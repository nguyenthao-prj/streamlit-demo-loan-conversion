import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict


# -----------------------------
# Helpers: numeric cleaning
# -----------------------------
NULL_TOKENS = {"", "none", "null", "nan", "n/a", "na", "None", "NULL", "NaN", "N/A"}

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Strip, remove commas, remove currency '$', coerce to numeric, invalid -> NaN."""
    if s.dtype.kind in "biufc":
        return s.astype(float)
    x = s.astype(str).str.strip()
    x = x.apply(lambda v: np.nan if v.strip().lower() in NULL_TOKENS else v)
    x = x.str.replace(",", "", regex=False)
    x = x.str.replace("$", "", regex=False)
    return pd.to_numeric(x, errors="coerce")


def parse_years_in_job(v: Any) -> float:
    """
    Report rule:
    - '10+ years' -> 11
    - '<1 year' -> 0
    - '2 years' -> 2
    """
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in NULL_TOKENS:
        return np.nan
    if "<1" in s:
        return 0.0
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    if "+" in s:
        return n + 1.0
    return n


@dataclass
class PreprocessSpec:
    # columns
    id_cols: Tuple[str, str] = ("Loan ID", "Customer ID")
    term_col: str = "Term" 
    purpose_col: str = "Purpose"
    home_col: str = "Home Ownership"

    categorical_features: Tuple[str, ...] = ("Term", "Home Ownership", "Purpose")
    job_years_col: str = "Years in current job"
    chargedoff_col: Optional[str] = "Loan Status"   # if exists; we will detect 'Charged Off'
    chargedoff_value: str = "Charged Off"

    # main numeric features (model inputs BEFORE derived features)
    base_numeric_features: Tuple[str, ...] = (
        "Current Loan Amount",
        "Credit Score",
        "Annual Income",
        "Years in current job",
        "Monthly Debt",
        "Years of Credit History",
        "Months since last delinquent",
        "Number of Open Accounts",
        "Number of Credit Problems",
        "Current Credit Balance",
        "Maximum Open Credit",
        "Bankruptcies",
        "Tax Liens"
    )

    # missing rules
    fill_zero_features: Tuple[str, ...] = (
        "Months since last delinquent",
        "Years in current job",
        "Bankruptcies",
        "Tax Liens",
    )
    hierarchical_impute_features: Tuple[str, ...] = ("Annual Income", "Credit Score")
    missing_indicators: Tuple[str, ...] = ("Annual Income", "Credit Score")

    # outlier placeholder rule
    loan_amount_placeholder: float = 99999999.0
    min_group_n: int = 30

    # derived features toggle (set True if your model was trained with them)
    add_derived_features: bool = False


class Preprocessor:
    """
    Production-lite preprocessor FIT on reference.csv:
    - stores group medians for hierarchical imputing
    - stores replacement medians for placeholder Current Loan Amount
    - stores global medians fallback
    - scaler (StandardScaler) will be attached later in build_preprocess.py
    """
    def __init__(self, spec: PreprocessSpec):
        self.spec = spec
        self.fitted_ = False

        # learned stats
        self.global_medians_: Dict[str, float] = {}
        self.group_medians_income_: Dict[Tuple, float] = {}
        self.group_medians_score_: Dict[Tuple, float] = {}
        self.loan_amt_repl_by_term_purpose_: Dict[Tuple[str, str], float] = {}
        self.loan_amt_repl_by_term_: Dict[str, float] = {}
        self.loan_amt_repl_global_: float = np.nan

        # hardening stats
        self._skipped_term_purpose_groups_ = 0

        # scaler will be attached later
        self.scaler_ = None
        self.final_feature_list_ = []
        # learned categories for stable one-hot columns
        self.home_categories_ = []
        self.purpose_categories_ = []


    def _normalize_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in [self.spec.term_col, self.spec.purpose_col, self.spec.home_col]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df

    def _clean_and_cast(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # parse Years in current job first
        if self.spec.job_years_col in df.columns:
            df[self.spec.job_years_col] = df[self.spec.job_years_col].apply(parse_years_in_job)

        # numeric cleaning for base features (except job_years already parsed)
        for c in self.spec.base_numeric_features:
            if c in df.columns and c != self.spec.job_years_col:
                df[c] = clean_numeric_series(df[c])

        return df

    def _make_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.spec.missing_indicators:
            if c in df.columns:
                df[f"{c}__missing"] = df[c].isna().astype(int)
        return df

    def _fill_zero(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.spec.fill_zero_features:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)
        return df

    def _replace_loan_amount_placeholder(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = "Current Loan Amount"
        if c not in df.columns:
            return df

        mask = df[c] == self.spec.loan_amount_placeholder
        if not mask.any():
            return df

        term = self.spec.term_col
        purp = self.spec.purpose_col

        for idx in df[mask].index:
            t = df.at[idx, term] if term in df.columns else None
            p = df.at[idx, purp] if purp in df.columns else None

            val = None
            if t is not None and p is not None and (t, p) in self.loan_amt_repl_by_term_purpose_:
                val = self.loan_amt_repl_by_term_purpose_[(t, p)]
            elif t is not None and t in self.loan_amt_repl_by_term_:
                val = self.loan_amt_repl_by_term_[t]
            else:
                val = self.loan_amt_repl_global_

            # final safety
            if val is None or np.isnan(val):
                val = self.loan_amt_repl_global_

            df.at[idx, c] = float(val)

        return df

    def _hierarchical_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hierarchical median imputation for Annual Income & Credit Score
        based on (Home Ownership, Purpose, Term) with fallback:
        (h,p,t) -> (p,t) -> (t) -> global median
        """
        df = df.copy()
        home, purp, term = self.spec.home_col, self.spec.purpose_col, self.spec.term_col

        for target in self.spec.hierarchical_impute_features:
            if target not in df.columns:
                continue

            is_missing = df[target].isna()
            if not is_missing.any():
                continue

            for idx in df[is_missing].index:
                h = df.at[idx, home] if home in df.columns else None
                p = df.at[idx, purp] if purp in df.columns else None
                t = df.at[idx, term] if term in df.columns else None

                val = None

                # (h,p,t)
                if (h is not None) and (p is not None) and (t is not None):
                    if target == "Annual Income":
                        val = self.group_medians_income_.get((h, p, t), None)
                    else:
                        val = self.group_medians_score_.get((h, p, t), None)

                # (p,t)
                if val is None and (p is not None) and (t is not None):
                    if target == "Annual Income":
                        val = self.group_medians_income_.get((None, p, t), None)
                    else:
                        val = self.group_medians_score_.get((None, p, t), None)

                # (t)
                if val is None and (t is not None):
                    if target == "Annual Income":
                        val = self.group_medians_income_.get((None, None, t), None)
                    else:
                        val = self.group_medians_score_.get((None, None, t), None)

                # global
                if val is None:
                    val = self.global_medians_.get(target, np.nan)

                # safety
                if val is None or np.isnan(val):
                    val = self.global_medians_.get(target, 0.0)

                df.at[idx, target] = float(val)

        return df

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df = self._normalize_categoricals(df)
        df = self._clean_and_cast(df)

        # Learn categorical levels from training data for stable one-hot
        if self.spec.home_col in df.columns:
            self.home_categories_ = sorted(
                df[self.spec.home_col].astype(str).str.strip().dropna().unique().tolist()
            )
        else:
            self.home_categories_ = []

        if self.spec.purpose_col in df.columns:
            self.purpose_categories_ = sorted(
                df[self.spec.purpose_col].astype(str).str.strip().dropna().unique().tolist()
            )
        else:
            self.purpose_categories_ = []


        # optional: drop Charged Off rows if exists
        if self.spec.chargedoff_col and self.spec.chargedoff_col in df.columns:
            df = df[df[self.spec.chargedoff_col].astype(str).str.strip() != self.spec.chargedoff_value].copy()

        # global medians for fallbacks
        for c in self.spec.base_numeric_features:
            if c in df.columns:
                self.global_medians_[c] = float(np.nanmedian(df[c].values))

        # loan amount placeholder replacement stats
        c = "Current Loan Amount"
        clean_df = df[df[c] != self.spec.loan_amount_placeholder].copy() if c in df.columns else df.copy()

        if c in clean_df.columns:
            self.loan_amt_repl_global_ = float(np.nanmedian(clean_df[c].values))

            # by term
            if self.spec.term_col in clean_df.columns:
                for t, g in clean_df.groupby(self.spec.term_col):
                    vals = g[c].values
                    valid_vals = vals[~np.isnan(vals)]
                    if len(valid_vals) == 0:
                        continue
                    self.loan_amt_repl_by_term_[t] = float(np.median(valid_vals))


            # by (term,purpose) with MIN_N and non-NaN median
            self._skipped_term_purpose_groups_ = 0
            if self.spec.term_col in clean_df.columns and self.spec.purpose_col in clean_df.columns:
                for (t, p), g in clean_df.groupby([self.spec.term_col, self.spec.purpose_col]):
                    if len(g) < self.spec.min_group_n:
                        continue
                    vals = g[c].values
                    valid_vals = vals[~np.isnan(vals)]
                    if len(valid_vals) < self.spec.min_group_n:
                        continue
                    med = float(np.median(valid_vals))
                    self.loan_amt_repl_by_term_purpose_[(t, p)] = med



        # hierarchical medians for income & score
        home, purp, term = self.spec.home_col, self.spec.purpose_col, self.spec.term_col
        for target in self.spec.hierarchical_impute_features:
            if target not in df.columns:
                continue

            # (h,p,t)
            if all(col in df.columns for col in [home, purp, term]):
                gfull = df.groupby([home, purp, term])[target].median()
                for (h, p, t), v in gfull.items():
                    if pd.isna(v):
                        continue
                    if target == "Annual Income":
                        self.group_medians_income_[(h, p, t)] = float(v)
                    else:
                        self.group_medians_score_[(h, p, t)] = float(v)

            # (p,t) fallback (store home=None)
            if all(col in df.columns for col in [purp, term]):
                gpt = df.groupby([purp, term])[target].median()
                for (p, t), v in gpt.items():
                    if pd.isna(v):
                        continue
                    if target == "Annual Income":
                        self.group_medians_income_[(None, p, t)] = float(v)
                    else:
                        self.group_medians_score_[(None, p, t)] = float(v)

            # (t) fallback (store home=None,purpose=None)
            if term in df.columns:
                gt = df.groupby([term])[target].median()
                for t, v in gt.items():
                    if pd.isna(v):
                        continue
                    if target == "Annual Income":
                        self.group_medians_income_[(None, None, t)] = float(v)
                    else:
                        self.group_medians_score_[(None, None, t)] = float(v)

        self.fitted_ = True
        return self
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ---- Term: binary encoding ----
        if self.spec.term_col in df.columns:
            x = df[self.spec.term_col].astype(str).str.strip()
            df["Term_Long Term"] = np.where(x == "Long Term", 1, 0).astype(int)
            df = df.drop(self.spec.term_col, axis=1)
        else:
            # ensure column exists if model trained with it
            df["Term_Long Term"] = 0

        # ---- Home Ownership: stable one-hot columns ----
        if self.spec.home_col in df.columns:
            x = df[self.spec.home_col].astype(str).str.strip()
            for cat in self.home_categories_:
                df[f"{self.spec.home_col}_{cat}"] = (x == cat).astype(int)
            df = df.drop(self.spec.home_col, axis=1)
        else:
            for cat in self.home_categories_:
                df[f"{self.spec.home_col}_{cat}"] = 0

        # ---- Purpose: stable one-hot columns ----
        if self.spec.purpose_col in df.columns:
            x = df[self.spec.purpose_col].astype(str).str.strip()
            for cat in self.purpose_categories_:
                df[f"{self.spec.purpose_col}_{cat}"] = (x == cat).astype(int)
            df = df.drop(self.spec.purpose_col, axis=1)
        else:
            for cat in self.purpose_categories_:
                df[f"{self.spec.purpose_col}_{cat}"] = 0

        return df

    def _fill_remaining_numeric_with_global_median(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.spec.base_numeric_features:
            if c in df.columns:
                med = self.global_medians_.get(c, np.nan)
                if med is None or np.isnan(med):
                    med = 0.0
                df[c] = df[c].fillna(float(med))
        return df


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Preprocessor chưa fit(). Hãy chạy build_preprocess.py trước.")
        out = df.copy()
        out = self._normalize_categoricals(out)
        out = self._make_missing_indicators(out)
        out = self._clean_and_cast(out)
        out = self._replace_loan_amount_placeholder(out)
        out = self._fill_zero(out)
        out = self._hierarchical_impute(out)
        out = self._fill_remaining_numeric_with_global_median(out)
        out = self._encode_categoricals(out)
        # Drop rows where all core numeric features are NaN
        core_feats = [
            "Current Loan Amount",
            "Monthly Debt",
            "Years of Credit History",
            "Number of Open Accounts",
            "Number of Credit Problems",
            "Current Credit Balance",
            "Maximum Open Credit",
        ]
        core_feats = [c for c in core_feats if c in out.columns]

        out = out.dropna(subset=core_feats, how="all")

        return out
    
    def get_feature_columns(self, df: pd.DataFrame):
        identifying_cols = ["Loan ID", "Customer ID", "Loan Status"]
        return [c for c in df.columns if c not in identifying_cols]

