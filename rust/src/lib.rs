//! Bootstrap TPR kernel: resamples scores and computes true positive rates at grid points.
//!
//! Per replicate: resample both classes with replacement; the TPR at grid
//! point `t` is the fraction of resampled positives strictly greater than
//! the resampled negatives' order statistic at descending 0-based index
//! `k_t`, where `k_t == n_neg` denotes a −∞ sentinel (TPR = 1).
//! Draws are tallied into count vectors over the pre-sorted originals and
//! thresholds/TPRs are read off with linear walks — O(n0 + n1 + K) per
//! replicate, O(n) memory per thread.
//!
//! Reproducibility contract: the RNG stream is a pure function of
//! `(seed, replicate_index)`, so output is bit-identical regardless of
//! thread count or scheduling.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        Self {
            s: [
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
            ],
        }
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Lemire bounded sampling; bias < n * 2^-64 (negligible, accepted).
    #[inline(always)]
    fn next_bounded(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// Derives a unique RNG seed for each replicate to ensure reproducible results across thread counts.
#[inline]
#[must_use]
pub fn replicate_seed(seed: u64, rep: u64) -> u64 {
    seed ^ rep.wrapping_mul(0xA24B_AED4_963E_E407)
}

/// One replicate. `neg_sorted`/`pos_sorted` ascending; `k_indices` ascending
/// with values in `[0, n0]`; `cnt_*` and `thresholds` are reused
/// thread-local buffers for efficiency.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn replicate(
    rng: &mut Xoshiro256pp,
    neg_sorted: &[f64],
    pos_sorted: &[f64],
    k_indices: &[u64],
    cnt_neg: &mut [u32],
    cnt_pos: &mut [u32],
    thresholds: &mut [f64],
    out_row: &mut [f64],
) {
    let n_neg = neg_sorted.len();
    let n_pos = pos_sorted.len();
    let n_grid = k_indices.len();

    cnt_neg.fill(0);
    cnt_pos.fill(0);
    for _ in 0..n_neg {
        cnt_neg[rng.next_bounded(n_neg)] += 1;
    }
    for _ in 0..n_pos {
        cnt_pos[rng.next_bounded(n_pos)] += 1;
    }

    // Thresholds: walk the sorted negatives from the top. After consuming
    // value i (descending), `cum` resampled elements occupy descending
    // positions [0, cum); grid point j is resolved once cum > k_j.
    // Grid points with k == n_neg use the -inf sentinel (TPR = 1); they sit
    // at the tail of the ascending k_indices array.
    let n_resolved = k_indices
        .iter()
        .take_while(|&&k| (k as usize) < n_neg)
        .count();
    let mut j = 0usize;
    let mut cum: u64 = 0;
    let mut i = n_neg;
    while j < n_resolved && i > 0 {
        i -= 1;
        let c = cnt_neg[i];
        if c == 0 {
            continue;
        }
        cum += u64::from(c);
        while j < n_resolved && k_indices[j] < cum {
            thresholds[j] = neg_sorted[i];
            j += 1;
        }
    }
    debug_assert_eq!(j, n_resolved);

    // TPR: thresholds are non-increasing in j, so a single backward pointer
    // over the sorted positives accumulates counts strictly above each one.
    let mut p = n_pos;
    let mut acc: u64 = 0;
    let inv_n_pos = 1.0f64 / n_pos as f64;
    for j in 0..n_resolved {
        let thr = thresholds[j];
        while p > 0 && pos_sorted[p - 1] > thr {
            acc += u64::from(cnt_pos[p - 1]);
            p -= 1;
        }
        out_row[j] = acc as f64 * inv_n_pos;
    }
    for j in n_resolved..n_grid {
        out_row[j] = 1.0;
    }
}

struct Buffers {
    cnt_neg: Vec<u32>,
    cnt_pos: Vec<u32>,
    thresholds: Vec<f64>,
}

fn compute_matrix(
    neg_sorted: &[f64],
    pos_sorted: &[f64],
    k_indices: &[u64],
    n_boot: usize,
    seed: u64,
) -> Vec<f64> {
    let n_grid = k_indices.len();
    let mut out = vec![0.0f64; n_boot * n_grid];
    out.par_chunks_mut(n_grid).enumerate().for_each_init(
        || Buffers {
            cnt_neg: vec![0u32; neg_sorted.len()],
            cnt_pos: vec![0u32; pos_sorted.len()],
            thresholds: vec![0.0f64; n_grid],
        },
        |buf, (rep, row)| {
            let mut rng = Xoshiro256pp::new(replicate_seed(seed, rep as u64));
            replicate(
                &mut rng,
                neg_sorted,
                pos_sorted,
                k_indices,
                &mut buf.cnt_neg,
                &mut buf.cnt_pos,
                &mut buf.thresholds,
                row,
            );
        },
    );
    out
}

/// Run the kernel, on a dedicated pool when `n_threads > 0`, else the
/// global rayon pool. Output is identical either way (replicate-indexed
/// RNG streams).
///
/// # Errors
///
/// Returns a message when any input is empty, `n_boot == 0`, `k_indices`
/// is out of range or not ascending, or the thread pool cannot be built.
pub fn bootstrap_tpr_matrix_vec(
    neg_sorted: &[f64],
    pos_sorted: &[f64],
    k_indices: &[u64],
    n_boot: usize,
    seed: u64,
    n_threads: usize,
) -> Result<Vec<f64>, String> {
    if neg_sorted.is_empty() || pos_sorted.is_empty() || k_indices.is_empty() || n_boot == 0 {
        return Err(format!(
            "empty input: n_neg={}, n_pos={}, n_grid={}, n_boot={n_boot} (all must be >= 1)",
            neg_sorted.len(),
            pos_sorted.len(),
            k_indices.len(),
        ));
    }
    let n_neg = neg_sorted.len() as u64;
    if k_indices.iter().any(|&k| k > n_neg) {
        return Err(format!("k_indices values must lie in [0, n_neg={n_neg}]"));
    }
    if k_indices.windows(2).any(|w| w[0] > w[1]) {
        return Err("k_indices must be ascending".to_string());
    }
    if n_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| e.to_string())?
            .install(|| {
                Ok(compute_matrix(
                    neg_sorted, pos_sorted, k_indices, n_boot, seed,
                ))
            })
    } else {
        Ok(compute_matrix(
            neg_sorted, pos_sorted, k_indices, n_boot, seed,
        ))
    }
}

/// Bootstrap TPR matrix `(n_boot, K)` where each row is a replicate.
// PyO3 extracts arguments by value; passing references is not an option here.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (neg_sorted, pos_sorted, k_indices, n_boot, seed, n_threads))]
fn bootstrap_tpr_matrix<'py>(
    py: Python<'py>,
    neg_sorted: PyReadonlyArray1<'py, f64>,
    pos_sorted: PyReadonlyArray1<'py, f64>,
    k_indices: PyReadonlyArray1<'py, u64>,
    n_boot: usize,
    seed: u64,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let neg = neg_sorted.as_slice()?.to_vec();
    let pos = pos_sorted.as_slice()?.to_vec();
    let ks = k_indices.as_slice()?.to_vec();
    let n_grid = ks.len();
    let data = py
        .allow_threads(|| bootstrap_tpr_matrix_vec(&neg, &pos, &ks, n_boot, seed, n_threads))
        .map_err(PyValueError::new_err)?;
    let arr = Array2::from_shape_vec((n_boot, n_grid), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bootstrap_tpr_matrix, m)?)?;
    Ok(())
}

#[cfg(test)]
// Exact float equality is the contract under test: kernel TPRs must be
// bit-identical to the oracle and across thread counts.
#[allow(clippy::float_cmp, clippy::cast_sign_loss)]
mod tests {
    use super::*;

    /// Brute-force oracle sharing the kernel's exact RNG stream: draw the
    /// resample lists with the same `next_bounded` calls, then sort and
    /// index literally. Uses the kernel's `count * (1/n_pos)` arithmetic so
    /// agreement is bit-exact.
    fn oracle_row(
        rng: &mut Xoshiro256pp,
        neg_sorted: &[f64],
        pos_sorted: &[f64],
        k_indices: &[u64],
    ) -> Vec<f64> {
        let n0 = neg_sorted.len();
        let n1 = pos_sorted.len();
        let mut neg_resamp: Vec<f64> = (0..n0).map(|_| neg_sorted[rng.next_bounded(n0)]).collect();
        let pos_resamp: Vec<f64> = (0..n1).map(|_| pos_sorted[rng.next_bounded(n1)]).collect();
        neg_resamp.sort_by(|a, b| b.partial_cmp(a).expect("finite scores"));
        let inv = 1.0f64 / n1 as f64;
        k_indices
            .iter()
            .map(|&k| {
                let thr = if (k as usize) == n0 {
                    f64::NEG_INFINITY
                } else {
                    neg_resamp[k as usize]
                };
                let cnt = pos_resamp.iter().filter(|&&x| x > thr).count();
                cnt as f64 * inv
            })
            .collect()
    }

    fn sorted_scores(rng: &mut Xoshiro256pp, n: usize, tie_every: usize) -> Vec<f64> {
        let mut v: Vec<f64> = (0..n)
            .map(|_| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
            .collect();
        if tie_every > 0 {
            for x in &mut v {
                *x = (*x * tie_every as f64).floor() / tie_every as f64;
            }
        }
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite scores"));
        v
    }

    fn k_grid(n_neg: usize, n_grid: usize) -> Vec<u64> {
        // linspace-style mapping incl. the k = 0 start and k = n_neg sentinel
        (0..n_grid)
            .map(|j| {
                let t = j as f64 / (n_grid - 1) as f64;
                (t * n_neg as f64).floor().min(n_neg as f64) as u64
            })
            .collect()
    }

    #[test]
    fn kernel_matches_bruteforce_oracle_small_inputs() {
        // n <= 64 across seeds, incl. all-ties, k=0/k=n_neg sentinels, n_neg=1
        let cases = [
            (1usize, 5usize, 0usize),
            (2, 2, 0),
            (7, 13, 0),
            (64, 64, 0),
            (16, 16, 4), // heavy ties
            (12, 8, 1),  // all scores tied at a single value
        ];
        for &(n0, n1, ties) in &cases {
            for seed in 0..8u64 {
                let mut gen_rng = Xoshiro256pp::new(seed.wrapping_add(999));
                let neg = sorted_scores(&mut gen_rng, n0, ties);
                let pos = sorted_scores(&mut gen_rng, n1, ties);
                let ks = k_grid(n0, n0 + 2);
                let n_boot = 32;
                let out =
                    bootstrap_tpr_matrix_vec(&neg, &pos, &ks, n_boot, seed, 1).expect("valid");
                for rep in 0..n_boot {
                    let mut rng = Xoshiro256pp::new(replicate_seed(seed, rep as u64));
                    let expect = oracle_row(&mut rng, &neg, &pos, &ks);
                    let got = &out[rep * ks.len()..(rep + 1) * ks.len()];
                    assert_eq!(
                        got,
                        expect.as_slice(),
                        "n0={n0} n1={n1} ties={ties} seed={seed} rep={rep}"
                    );
                }
            }
        }
    }

    #[test]
    fn output_is_independent_of_thread_count() {
        let mut gen_rng = Xoshiro256pp::new(7);
        let neg = sorted_scores(&mut gen_rng, 200, 0);
        let pos = sorted_scores(&mut gen_rng, 150, 0);
        let ks = k_grid(200, 64);
        let reference = bootstrap_tpr_matrix_vec(&neg, &pos, &ks, 500, 42, 1).expect("valid");
        for threads in [2usize, 4, 0] {
            let out = bootstrap_tpr_matrix_vec(&neg, &pos, &ks, 500, 42, threads).expect("valid");
            assert_eq!(out, reference, "thread count {threads} changed the output");
        }
    }

    #[test]
    fn sentinel_column_pins_tpr_to_one() {
        let mut gen_rng = Xoshiro256pp::new(3);
        let neg = sorted_scores(&mut gen_rng, 10, 0);
        let pos = sorted_scores(&mut gen_rng, 10, 0);
        let ks = vec![0u64, 5, 10];
        let out = bootstrap_tpr_matrix_vec(&neg, &pos, &ks, 64, 1, 1).expect("valid");
        for rep in 0..64 {
            assert!((out[rep * 3 + 2] - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn rejects_empty_and_out_of_range_inputs() {
        let neg = vec![0.0, 1.0];
        let pos = vec![0.5];
        assert!(bootstrap_tpr_matrix_vec(&[], &pos, &[0], 1, 0, 1).is_err());
        assert!(bootstrap_tpr_matrix_vec(&neg, &[], &[0], 1, 0, 1).is_err());
        assert!(bootstrap_tpr_matrix_vec(&neg, &pos, &[], 1, 0, 1).is_err());
        assert!(bootstrap_tpr_matrix_vec(&neg, &pos, &[0], 0, 0, 1).is_err());
        assert!(bootstrap_tpr_matrix_vec(&neg, &pos, &[3], 1, 0, 1).is_err()); // k > n_neg
        assert!(bootstrap_tpr_matrix_vec(&neg, &pos, &[1, 0], 1, 0, 1).is_err());
        // not ascending
    }

    #[test]
    fn all_ties_input_gives_step_at_sentinel_only() {
        // every score equal: strictly-greater count is always 0 for real
        // thresholds, 1.0 only at the sentinel
        let neg = vec![0.5; 8];
        let pos = vec![0.5; 8];
        let ks = vec![0u64, 4, 8];
        let out = bootstrap_tpr_matrix_vec(&neg, &pos, &ks, 16, 9, 1).expect("valid");
        for rep in 0..16 {
            assert_eq!(out[rep * 3], 0.0);
            assert_eq!(out[rep * 3 + 1], 0.0);
            assert_eq!(out[rep * 3 + 2], 1.0);
        }
    }
}
