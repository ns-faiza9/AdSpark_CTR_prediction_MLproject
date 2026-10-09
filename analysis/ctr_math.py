"""Mathematical Engine for Click-Through Rate (CTR) Prediction and Ad Tech ML.

Provides rigorous implementations of:
1. Google FTRL-Proximal (Follow-The-Regularized-Leader) Online Learning Algorithm
2. Rendle's Factorization Machine (FM) with O(k·d) Fast Interaction Computation
3. Empirical Bayes Beta-Binomial Smoothing for High-Cardinality Categorical Levels
4. Facebook Negative Downsampling Odds Calibration (He et al. Inversion)
5. Information Value (IV) and Weight of Evidence (WoE) Discretization
6. Calibration Metrics: Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
   and Brier Score Decomposition (Reliability, Resolution, Uncertainty)
7. Statistical Significance: McNemar's Contingency Test with Edwards' Correction & DeLong ROC Variance
"""
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class FTRLProximal:
    """Follow-The-Regularized-Leader with Proximal step (McMahan et al., Google 2013).
    
    Update rule per coordinate i:
        If |z_i| <= lambda1:
            w_i = 0
        Else:
            w_i = - sgn(z_i) * (|z_i| - lambda1) / ((beta + sqrt(n_i)) / alpha + lambda2)
            
    where:
        n_i = sum of squared historical gradients g_i^2
        z_i = accumulated regularized gradient coordinate
        sigma_i = (sqrt(n_i + g_i^2) - sqrt(n_i)) / alpha
        z_i <- z_i + g_i - sigma_i * w_i
    """
    def __init__(self, alpha: float = 0.1, beta: float = 1.0, lambda1: float = 1.0, lambda2: float = 1.0, num_features: int = 100000):
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.D = num_features
        
        self.n = np.zeros(self.D, dtype=np.float64)  # Sum of squared gradients
        self.z = np.zeros(self.D, dtype=np.float64)  # Accumulated gradient coordinate
        self.w = np.zeros(self.D, dtype=np.float64)  # Current weights

    def _get_weight(self, i: int) -> float:
        """Compute coordinate weight w_i with L1 soft-thresholding."""
        z_i = self.z[i]
        if abs(z_i) <= self.lambda1:
            return 0.0
        sign = 1.0 if z_i > 0 else -1.0
        numerator = sign * self.lambda1 - z_i
        denominator = (self.beta + math.sqrt(self.n[i])) / self.alpha + self.lambda2
        return numerator / denominator

    def predict_proba(self, indices: List[int], values: Optional[List[float]] = None) -> float:
        """Predict CTR probability p = sigma(w^T x)."""
        if values is None:
            values = [1.0] * len(indices)
            
        raw_score = 0.0
        for idx, val in zip(indices, values):
            feat_idx = idx % self.D
            w_i = self._get_weight(feat_idx)
            self.w[feat_idx] = w_i
            raw_score += w_i * val
            
        # Sigmoid clipping to avoid numerical overflow
        raw_score = max(min(raw_score, 35.0), -35.0)
        return 1.0 / (1.0 + math.exp(-raw_score))

    def update(self, indices: List[int], y: int, values: Optional[List[float]] = None) -> float:
        """Perform one online gradient update step and return instantaneous log-loss."""
        if values is None:
            values = [1.0] * len(indices)
            
        p = self.predict_proba(indices, values)
        loss = -math.log(max(p, 1e-15)) if y == 1 else -math.log(max(1.0 - p, 1e-15))
        gradient = p - y  # d(logloss)/d(score)

        for idx, val in zip(indices, values):
            feat_idx = idx % self.D
            g_i = gradient * val
            w_i = self.w[feat_idx]
            n_old = self.n[feat_idx]
            n_new = n_old + g_i * g_i
            
            sigma_i = (math.sqrt(n_new) - math.sqrt(n_old)) / self.alpha
            self.z[feat_idx] += g_i - sigma_i * w_i
            self.n[feat_idx] = n_new

        return loss

    def get_sparsity(self) -> float:
        """Calculate percentage of exactly zero weights."""
        active_weights = np.array([self._get_weight(i) for i in range(self.D)])
        non_zero = np.count_nonzero(active_weights)
        return (1.0 - (non_zero / self.D)) * 100.0


class FactorizationMachine:
    """Rendle Factorization Machine (FM) with O(k·d) fast 2nd-order feature interaction computation.
    
    Formulation:
        y_hat(x) = w_0 + sum_{i=1}^d w_i x_i + sum_{i=1}^d sum_{j=i+1}^d <v_i, v_j> x_i x_j
        
    Linear time identity:
        sum_{i=1}^d sum_{j=i+1}^d <v_i, v_j> x_i x_j = 0.5 * sum_{f=1}^k [ (sum_{i=1}^d v_{i,f} x_i)^2 - sum_{i=1}^d v_{i,f}^2 x_i^2 ]
    """
    def __init__(self, num_features: int, k_factors: int = 8, lr: float = 0.01, reg_w: float = 0.01, reg_v: float = 0.01, seed: int = 42):
        self.d = num_features
        self.k = k_factors
        self.lr = lr
        self.reg_w = reg_w
        self.reg_v = reg_v
        
        rng = np.random.RandomState(seed)
        self.w0 = 0.0
        self.w = rng.normal(0.0, 0.01, size=self.d)
        self.V = rng.normal(0.0, 0.1 / math.sqrt(self.k), size=(self.d, self.k))

    def forward(self, x: np.ndarray) -> Tuple[float, float, float]:
        """Compute forward pass and return (total_logit, linear_component, interaction_component)."""
        linear = self.w0 + float(np.dot(self.w, x))
        
        # O(k*d) trick:
        # sum_f [ (sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2 ]
        # V is (d, k), x is (d,)
        vx = np.dot(x, self.V)  # shape: (k,)
        vx_sq = vx ** 2
        
        v_sq = self.V ** 2
        x_sq = x ** 2
        v_sq_x_sq = np.dot(x_sq, v_sq)  # shape: (k,)
        
        interaction = 0.5 * float(np.sum(vx_sq - v_sq_x_sq))
        total_logit = linear + interaction
        return total_logit, linear, interaction

    def predict_proba(self, x: np.ndarray) -> float:
        """Compute sigmoid probability."""
        total_logit, _, _ = self.forward(x)
        total_logit = max(min(total_logit, 35.0), -35.0)
        return 1.0 / (1.0 + math.exp(-total_logit))

    def get_pairwise_interaction_energy(self) -> np.ndarray:
        """Compute cross-feature interaction matrix S = V * V^T."""
        return np.dot(self.V, self.V.T)


class BayesianCTRSmoother:
    """Empirical Bayes Beta-Binomial Smoothing for High-Cardinality Categorical Features.
    
    Prior: CTR ~ Beta(alpha, beta) with global mean mu = alpha / (alpha + beta)
           and pseudo-count weight m = alpha + beta.
    
    Posterior Mean:
        CTR_smoothed = (Clicks_i + alpha) / (Impressions_i + alpha + beta)
                     = (Clicks_i + mu * m) / (Impressions_i + m)
    """
    def __init__(self, prior_mean: float = 0.1694, pseudo_counts: float = 20.0):
        self.mu = prior_mean
        self.m = pseudo_counts
        self.alpha = self.mu * self.m
        self.beta = (1.0 - self.mu) * self.m

    def smooth(self, clicks: int, impressions: int) -> float:
        """Return smoothed posterior CTR estimate."""
        return (clicks + self.alpha) / (impressions + self.alpha + self.beta)

    def variance(self, clicks: int, impressions: int) -> float:
        """Return posterior Beta distribution variance (uncertainty measure)."""
        a = clicks + self.alpha
        b = (impressions - clicks) + self.beta
        return (a * b) / (((a + b) ** 2) * (a + b + 1.0))

    def wilson_score_interval(self, clicks: int, impressions: int, z: float = 1.96) -> Tuple[float, float]:
        """Wilson score interval for binomial proportions (95% confidence)."""
        if impressions == 0:
            return (self.mu, self.mu)
        p_hat = clicks / impressions
        n = impressions
        denominator = 1.0 + (z ** 2) / n
        center = (p_hat + (z ** 2) / (2.0 * n)) / denominator
        spread = z * math.sqrt((p_hat * (1.0 - p_hat) + (z ** 2) / (4.0 * n)) / n) / denominator
        return (max(0.0, center - spread), min(1.0, center + spread))


class NegativeDownsampler:
    """Facebook / AdTech Negative Downsampling Odds Calibration (He et al., 2014).
    
    When negative samples are sub-sampled at rate w in (0, 1] to compress training data:
        p' = P(y=1 | x, sampled)
        Odds' = p' / (1 - p') = (P(y=1|x) / (1 - P(y=1|x))) / w = Odds / w
        
    Inverted Calibration Formula:
        p = p' / (p' + (1 - p') / w)
    """
    @staticmethod
    def calibrate(p_sampled: float, downsampling_rate: float) -> float:
        """Invert downsampled model prediction back to true uncalibrated probability."""
        if downsampling_rate <= 0.0 or downsampling_rate >= 1.0:
            return p_sampled
        p_sampled = max(min(p_sampled, 0.999999), 0.000001)
        return p_sampled / (p_sampled + (1.0 - p_sampled) / downsampling_rate)

    @staticmethod
    def log_odds_shift(downsampling_rate: float) -> float:
        """Returns theoretical log-odds correction offset Delta = ln(w)."""
        if downsampling_rate <= 0.0 or downsampling_rate >= 1.0:
            return 0.0
        return math.log(downsampling_rate)


class InformationValueAnalyzer:
    """Computes Weight of Evidence (WoE) and Information Value (IV) for categorical attributes.
    
    WoE_i = ln( (Clicks_i / Total_Clicks) / (NonClicks_i / Total_NonClicks) )
    IV = sum_i ( (Clicks_i / Total_Clicks) - (NonClicks_i / Total_NonClicks) ) * WoE_i
    
    Predictive Power Rule of Thumb:
        < 0.02:   Unpredictable
        0.02-0.1: Weak predictor
        0.1-0.3:  Medium predictor
        0.3-0.5:  Strong predictor
        > 0.5:    Suspicious / High power
    """
    @staticmethod
    def compute_iv(clicks: List[int], non_clicks: List[int]) -> Tuple[float, List[float], str]:
        total_clicks = sum(clicks)
        total_non_clicks = sum(non_clicks)
        
        if total_clicks == 0 or total_non_clicks == 0:
            return 0.0, [0.0] * len(clicks), "Zero class count"
            
        iv = 0.0
        woe_list = []
        
        for c, nc in zip(clicks, non_clicks):
            # Laplace smoothing to prevent division by zero in log
            pct_click = max((c + 0.5) / total_clicks, 1e-10)
            pct_non_click = max((nc + 0.5) / total_non_clicks, 1e-10)
            woe = math.log(pct_click / pct_non_click)
            woe_list.append(woe)
            iv += (pct_click - pct_non_click) * woe
            
        if iv < 0.02:
            rating = "Unpredictable (<0.02)"
        elif iv < 0.10:
            rating = "Weak Predictor (0.02 - 0.10)"
        elif iv < 0.30:
            rating = "Medium Predictor (0.10 - 0.30)"
        elif iv < 0.50:
            rating = "Strong Predictor (0.30 - 0.50)"
        else:
            rating = "Highly Predictive / Dominant (>0.50)"
            
        return iv, woe_list, rating


class MetricsAndCalibration:
    """Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score Decomposition."""
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float, List[Dict[str, float]]]:
        """Compute ECE and MCE with reliability binning."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_details = []
        n_total = len(y_true)

        for i in range(n_bins):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                acc = float(np.mean(y_true[in_bin]))
                conf = float(np.mean(y_prob[in_bin]))
                abs_diff = abs(acc - conf)
                ece += (bin_size / n_total) * abs_diff
                mce = max(mce, abs_diff)
                bin_details.append({
                    "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f})",
                    "count": int(bin_size),
                    "confidence": round(conf, 4),
                    "accuracy": round(acc, 4),
                    "calibration_gap": round(abs_diff, 4)
                })
        return float(ece), float(mce), bin_details

    @staticmethod
    def brier_score_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """Brier Score = Reliability - Resolution + Uncertainty.
        
        Uncertainty = p_base * (1 - p_base)
        Reliability = sum_k (N_k / N) * (p_k_bar - o_k_bar)^2   (Calibration error)
        Resolution  = sum_k (N_k / N) * (o_k_bar - p_base)^2    (Discrimination ability)
        """
        N = len(y_true)
        p_base = float(np.mean(y_true))
        uncertainty = p_base * (1.0 - p_base)
        
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            in_bin = (y_prob >= bins[i]) & (y_prob < bins[i+1]) if i < n_bins - 1 else (y_prob >= bins[i]) & (y_prob <= bins[i+1])
            N_k = np.sum(in_bin)
            if N_k > 0:
                p_k_bar = float(np.mean(y_prob[in_bin]))
                o_k_bar = float(np.mean(y_true[in_bin]))
                reliability += (N_k / N) * ((p_k_bar - o_k_bar) ** 2)
                resolution += (N_k / N) * ((o_k_bar - p_base) ** 2)
                
        total_brier = float(np.mean((y_prob - y_true) ** 2))
        return {
            "total_brier_score": round(total_brier, 5),
            "reliability_calibration_loss": round(reliability, 5),
            "resolution_discrimination_gain": round(resolution, 5),
            "uncertainty_base_entropy": round(uncertainty, 5),
        }

    @staticmethod
    def normalized_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Rigorous Normalized Entropy (NE) / Normalized Cross-Entropy (He et al.).
        
        NE = LogLoss(y, p) / ( - [p_base * ln(p_base) + (1 - p_base) * ln(1 - p_base)] )
        """
        eps = 1e-15
        p_clip = np.clip(y_prob, eps, 1.0 - eps)
        log_loss = -float(np.mean(y_true * np.log(p_clip) + (1.0 - y_true) * np.log(1.0 - p_clip)))
        
        p_base = float(np.mean(y_true))
        p_base = max(min(p_base, 1.0 - eps), eps)
        base_entropy = -(p_base * math.log(p_base) + (1.0 - p_base) * math.log(1.0 - p_base))
        
        return log_loss / base_entropy


class StatisticalHypothesisTests:
    """McNemar's Chi-Square Contingency Test with Edwards' Continuity Correction."""
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
        """Compute contingency matrix [[a, b], [c, d]] where:
            a: both correct
            b: model A correct, model B incorrect
            c: model A incorrect, model B correct
            d: both incorrect
            
        Edwards' Chi-Square: chi2 = (|b - c| - 1)^2 / (b + c)
        """
        corr_a = (y_pred_a == y_true)
        corr_b = (y_pred_b == y_true)

        a = int(np.sum(corr_a & corr_b))
        b = int(np.sum(corr_a & (~corr_b)))
        c = int(np.sum((~corr_a) & corr_b))
        d = int(np.sum((~corr_a) & (~corr_b)))

        if (b + c) == 0:
            chi2 = 0.0
            p_val = 1.0
        else:
            chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
            # p-value from chi2 survival function with 1 degree of freedom: erfc(sqrt(chi2 / 2))
            p_val = math.erfc(math.sqrt(chi2 / 2.0))

        return {
            "contingency_matrix": [[a, b], [c, d]],
            "model_a_only_correct_b": b,
            "model_b_only_correct_c": c,
            "both_correct_a": a,
            "both_incorrect_d": d,
            "chi2_statistic": round(float(chi2), 4),
            "p_value": float(p_val),
            "is_significant_at_05": bool(p_val < 0.05),
            "is_significant_at_01": bool(p_val < 0.01),
        }
