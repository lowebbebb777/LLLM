"""numeric_stats_eval 評価セットの生成 + 検証 (SPEC §5.2)。

    python3 scripts/gen_eval.py            # 検証し data/numeric_stats_eval/problems.jsonl を書き出す
    python3 scripts/gen_eval.py --check    # 書き出さず検証のみ (CI 用)


方針 (前回の反省: 未検証の値をコミットしない):
  - numeric: expected は Python で計算した値をそのまま埋める (手打ちしない)。
  - code   : test を参照実装 _ref で実行し pass を確認。さらに誤実装 _wrong で
             fail することを確認 (弱い/自明なテストを排除)。
  - 最後に evaluate.py のオラクルで numeric overall==1.0, path agreement==1.0,
             dispersion==0.0 を確認してから jsonl を書き出す。
"""
import json
import math
import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from evaluate import (  # noqa: E402
    check_correctness,
    load_numeric_problems,
    numeric_consistency_score,
    path_independence_score,
)

PROBLEMS = []          # 出力用 dict のリスト
_CODE_CHECKS = []      # (id, ref, wrong, test) 検証用


def num(pid, category, prompt, expected, rel_tol, paraphrases):
    """numeric 問題を登録 (expected は呼び出し側で計算済みの値)。"""
    assert len(paraphrases) >= 2, f"{pid}: numeric は paraphrases 2 個以上 (§5.2b)"
    PROBLEMS.append({
        "id": pid, "category": category, "kind": "numeric", "prompt": prompt,
        "expected": round(float(expected), 6), "rel_tol": rel_tol,
        "paraphrases": paraphrases,
    })


def code(pid, category, prompt, test, ref, wrong):
    """code 問題を登録。ref/wrong は検証専用 (jsonl には出さない)。"""
    PROBLEMS.append({
        "id": pid, "category": category, "kind": "code",
        "prompt": prompt, "test": test,
    })
    _CODE_CHECKS.append((pid, ref, wrong, test))


# ===========================================================================
# cpk: プロセス能力指数
# ===========================================================================
def cp(usl, lsl, sigma):
    return (usl - lsl) / (6 * sigma)


def cpk(usl, lsl, mean, sigma):
    return min((usl - mean) / (3 * sigma), (mean - lsl) / (3 * sigma))


def cpu(usl, mean, sigma):
    return (usl - mean) / (3 * sigma)


def cpl(lsl, mean, sigma):
    return (mean - lsl) / (3 * sigma)


num("cpk_001", "cpk",
    "A process has USL=10, LSL=2, mean=6, sigma=1. Compute Cpk. Answer with the numeric value only.",
    cpk(10, 2, 6, 1), 0.01,
    ["For a manufacturing process where the upper spec limit is 10, the lower spec limit is 2, the mean is 6 and the standard deviation is 1, what is the process capability index Cpk? Give only the number.",
     "Given sigma=1, mu=6, with tolerance band [2, 10], calculate Cpk and report just the value.",
     "Cpk = min((USL-mu)/(3*sigma), (mu-LSL)/(3*sigma)). With USL=10, LSL=2, mu=6, sigma=1, evaluate it. Number only."])
num("cp_001", "cpk",
    "A process has USL=10, LSL=2, sigma=1. Compute Cp (process potential). Answer with the numeric value only.",
    cp(10, 2, 1), 0.01,
    ["With specification limits 2 and 10 and a standard deviation of 1, what is Cp? Number only.",
     "Cp = (USL - LSL) / (6*sigma). For USL=10, LSL=2, sigma=1, give the value only."])
num("cpk_002", "cpk",
    "A process has USL=8, LSL=2, mean=5, sigma=1. Compute Cpk. Answer with the numeric value only.",
    cpk(8, 2, 5, 1), 0.01,
    ["Upper spec 8, lower spec 2, process mean 5, sigma 1: what is Cpk? Value only.",
     "Evaluate min((8-5)/3, (5-2)/3). Number only."])
num("cpk_003", "cpk",
    "A process has USL=10, LSL=2, mean=4, sigma=1. Compute Cpk (note the mean is off-center). Answer with the numeric value only.",
    cpk(10, 2, 4, 1), 0.01,
    ["USL=10, LSL=2, mu=4, sigma=1. The mean is shifted low. Give Cpk only.",
     "With a low-shifted mean of 4 (limits 2 and 10, sigma 1), the capability is limited by the lower side. Compute Cpk. Number only."])
num("cpk_004", "cpk",
    "A process has USL=20, LSL=8, mean=14, sigma=2. Compute Cpk. Answer with the numeric value only.",
    cpk(20, 8, 14, 2), 0.01,
    ["Limits 8 and 20, centered mean 14, sigma 2: compute Cpk. Value only.",
     "min((20-14)/(3*2), (14-8)/(3*2)) = ? Number only."])
num("cpk_005", "cpk",
    "A process has USL=100, LSL=40, mean=70, sigma=5. Compute Cpk. Answer with the numeric value only.",
    cpk(100, 40, 70, 5), 0.01,
    ["USL 100, LSL 40, mean 70, sigma 5. What is Cpk? Number only.",
     "A well-centered process on [40,100] with sigma 5. Give its Cpk value only."])
num("cp_002", "cpk",
    "A process has USL=14, LSL=2, sigma=2. Compute Cp. Answer with the numeric value only.",
    cp(14, 2, 2), 0.01,
    ["Spec width from 2 to 14, sigma 2. What is Cp? Value only.",
     "(14 - 2) / (6 * 2) = ? Number only."])
num("cp_003", "cpk",
    "A process has USL=0.55, LSL=0.45, sigma=0.01. Compute Cp. Answer with the numeric value only.",
    cp(0.55, 0.45, 0.01), 0.01,
    ["Tight tolerance [0.45, 0.55] with sigma 0.01. Compute Cp. Value only.",
     "Cp for spec limits 0.45 and 0.55 and standard deviation 0.01. Number only."])
num("cpu_001", "cpk",
    "A process has USL=10, mean=7, sigma=1. Compute the one-sided upper capability Cpu=(USL-mean)/(3*sigma). Answer with the numeric value only.",
    cpu(10, 7, 1), 0.01,
    ["Upper spec 10, mean 7, sigma 1. What is Cpu (upper one-sided capability)? Value only.",
     "(10 - 7) / (3 * 1) = ? Number only."])
num("cpl_001", "cpk",
    "A process has LSL=2, mean=5, sigma=1. Compute the one-sided lower capability Cpl=(mean-LSL)/(3*sigma). Answer with the numeric value only.",
    cpl(2, 5, 1), 0.01,
    ["Lower spec 2, mean 5, sigma 1. What is Cpl (lower one-sided capability)? Value only.",
     "(5 - 2) / (3 * 1) = ? Number only."])

code("cpk_code_001", "cpk",
     "Write a Python function `cpk(usl, lsl, mean, sigma)` that returns the process capability index Cpk.",
     "assert abs(cpk(10, 2, 6, 1) - 1.3333) < 1e-3\n"
     "assert abs(cpk(8, 2, 5, 1) - 1.0) < 1e-3\n"
     "assert abs(cpk(10, 2, 4, 1) - 0.6667) < 1e-3",
     "def cpk(usl, lsl, mean, sigma):\n    return min((usl-mean)/(3*sigma), (mean-lsl)/(3*sigma))",
     "def cpk(usl, lsl, mean, sigma):\n    return (usl-lsl)/(6*sigma)")  # Cp を返す誤り
code("cp_code_001", "cpk",
     "Write a Python function `cp(usl, lsl, sigma)` that returns the process potential index Cp = (USL-LSL)/(6*sigma).",
     "assert abs(cp(10, 2, 1) - 1.3333) < 1e-3\n"
     "assert abs(cp(14, 2, 2) - 1.0) < 1e-3\n"
     "assert abs(cp(0.55, 0.45, 0.01) - 1.6667) < 1e-3",
     "def cp(usl, lsl, sigma):\n    return (usl-lsl)/(6*sigma)",
     "def cp(usl, lsl, sigma):\n    return (usl-lsl)/(3*sigma)")
code("cpu_code_001", "cpk",
     "Write a Python function `cpu(usl, mean, sigma)` returning the upper one-sided capability (USL-mean)/(3*sigma).",
     "assert abs(cpu(10, 7, 1) - 1.0) < 1e-6\n"
     "assert abs(cpu(10, 4, 1) - 2.0) < 1e-6",
     "def cpu(usl, mean, sigma):\n    return (usl-mean)/(3*sigma)",
     "def cpu(usl, mean, sigma):\n    return (usl-mean)/sigma")
code("cpk_code_002", "cpk",
     "Write a Python function `capability(usl, lsl, mean, sigma)` returning a tuple (Cp, Cpk).",
     "cp_, cpk_ = capability(10, 2, 6, 1)\n"
     "assert abs(cp_ - 1.3333) < 1e-3 and abs(cpk_ - 1.3333) < 1e-3\n"
     "cp2, cpk2 = capability(10, 2, 4, 1)\n"
     "assert abs(cp2 - 1.3333) < 1e-3 and abs(cpk2 - 0.6667) < 1e-3",
     "def capability(usl, lsl, mean, sigma):\n"
     "    cp = (usl-lsl)/(6*sigma)\n"
     "    cpk = min((usl-mean)/(3*sigma), (mean-lsl)/(3*sigma))\n"
     "    return (cp, cpk)",
     "def capability(usl, lsl, mean, sigma):\n"
     "    cp = (usl-lsl)/(6*sigma)\n"
     "    return (cp, cp)")  # Cpk=Cp とする誤り


# ===========================================================================
# stats_impl: 統計手法の実装
# ===========================================================================
D = [2, 4, 4, 4, 5, 5, 7, 9]  # mean 5, pop var 4, pop std 2


def pop_std(xs):
    return statistics.pstdev(xs)


def samp_std(xs):
    return statistics.stdev(xs)


def pearson(xs, ys):
    return statistics.correlation(xs, ys)


def ols_slope(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num_ = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = sum((x-mx)**2 for x in xs)
    return num_/den


PXY_X = [1, 2, 3, 4, 5]
PXY_Y = [2, 4, 5, 4, 5]

num("mean_001", "stats_impl",
    "Compute the arithmetic mean of [10, 20, 30]. Answer with the numeric value only.",
    statistics.fmean([10, 20, 30]), 1e-4,
    ["What is the average of 10, 20 and 30? Number only.",
     "Sum 10+20+30 and divide by the count. Give just the result."])
num("pop_std_001", "stats_impl",
    "Compute the population standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]. Answer with the numeric value only.",
    pop_std(D), 0.005,
    ["What is the population (not sample) standard deviation of the dataset 2,4,4,4,5,5,7,9? Number only.",
     "Given the numbers [2,4,4,4,5,5,7,9], report sigma_population. Just the value."])
num("samp_std_001", "stats_impl",
    "Compute the SAMPLE standard deviation (n-1 denominator) of [2, 4, 4, 4, 5, 5, 7, 9]. Answer with the numeric value only.",
    samp_std(D), 0.01,
    ["What is the sample standard deviation (Bessel-corrected, divide by n-1) of 2,4,4,4,5,5,7,9? Number only.",
     "Using the n-1 formula, compute the standard deviation of [2,4,4,4,5,5,7,9]. Value only."])
num("pop_var_001", "stats_impl",
    "Compute the population variance of [2, 4, 4, 4, 5, 5, 7, 9]. Answer with the numeric value only.",
    statistics.pvariance(D), 0.005,
    ["What is the population variance of 2,4,4,4,5,5,7,9? Number only.",
     "Mean-squared-deviation (population) of [2,4,4,4,5,5,7,9]. Value only."])
num("median_001", "stats_impl",
    "Compute the median of [3, 1, 4, 1, 5, 9, 2, 6]. Answer with the numeric value only.",
    statistics.median([3, 1, 4, 1, 5, 9, 2, 6]), 1e-4,
    ["What is the median of the dataset 3,1,4,1,5,9,2,6? Number only.",
     "Sort 3,1,4,1,5,9,2,6 and take the middle (average of the two central values). Value only."])
num("cv_001", "stats_impl",
    "Compute the coefficient of variation (population std / mean) of [2, 4, 4, 4, 5, 5, 7, 9]. Answer with the numeric value only.",
    pop_std(D)/statistics.fmean(D), 0.01,
    ["What is the CV (sigma over mean, population) of 2,4,4,4,5,5,7,9? Number only.",
     "Divide the population standard deviation by the mean for [2,4,4,4,5,5,7,9]. Value only."])
num("zscore_val_001", "stats_impl",
    "For the dataset [2, 4, 4, 4, 5, 5, 7, 9], compute the population z-score of the value 9. Answer with the numeric value only.",
    (9 - statistics.fmean(D))/pop_std(D), 0.01,
    ["In 2,4,4,4,5,5,7,9, how many population standard deviations above the mean is 9? Number only.",
     "z = (9 - mean) / sigma_pop for the dataset [2,4,4,4,5,5,7,9]. Value only."])
num("range_001", "stats_impl",
    "Compute the range (max - min) of [2, 4, 4, 4, 5, 5, 7, 9]. Answer with the numeric value only.",
    max(D) - min(D), 1e-4,
    ["What is the range (largest minus smallest) of 2,4,4,4,5,5,7,9? Number only.",
     "Subtract the minimum from the maximum of [2,4,4,4,5,5,7,9]. Value only."])
num("geomean_001", "stats_impl",
    "Compute the geometric mean of [1, 2, 4]. Answer with the numeric value only.",
    (1*2*4)**(1/3), 0.01,
    ["What is the geometric mean (cube root of the product) of 1, 2 and 4? Number only.",
     "Geometric mean of [1,2,4] = (1*2*4)**(1/3). Value only."])
num("pearson_001", "stats_impl",
    "Compute the Pearson correlation coefficient between xs=[1,2,3,4,5] and ys=[2,4,5,4,5]. Answer with the numeric value only.",
    pearson(PXY_X, PXY_Y), 0.01,
    ["What is Pearson's r for the paired data x=1,2,3,4,5 and y=2,4,5,4,5? Number only.",
     "Correlation coefficient between [1,2,3,4,5] and [2,4,5,4,5]. Value only."])
num("ols_slope_num_001", "stats_impl",
    "Compute the OLS regression slope of ys on xs for xs=[1,2,3,4,5], ys=[2,4,5,4,5]. Answer with the numeric value only.",
    ols_slope(PXY_X, PXY_Y), 0.01,
    ["Fit y = a + b*x by least squares to x=1,2,3,4,5 and y=2,4,5,4,5; report the slope b only.",
     "What is the least-squares slope for the points (1,2),(2,4),(3,5),(4,4),(5,5)? Number only."])

code("ols_code_001", "stats_impl",
     "Write a Python function `ols_slope(xs, ys)` that returns the ordinary least squares regression slope of ys on xs.",
     "assert abs(ols_slope([1,2,3,4], [2,4,6,8]) - 2.0) < 1e-6\n"
     "assert abs(ols_slope([0,1,2], [1,3,5]) - 2.0) < 1e-6\n"
     "assert abs(ols_slope([1,2,3], [3,2,1]) + 1.0) < 1e-6\n"
     "assert abs(ols_slope([1,2,3,4], [1,2,3,10]) - 2.8) < 1e-6",  # 非直線: 端点勾配(3.0)と分離
     "def ols_slope(xs, ys):\n"
     "    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n\n"
     "    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))\n"
     "    den=sum((x-mx)**2 for x in xs)\n"
     "    return num/den",
     "def ols_slope(xs, ys):\n    return (ys[-1]-ys[0])/(xs[-1]-xs[0])")  # 端点勾配 (非最小二乗)
code("zscore_code_001", "stats_impl",
     "Write a Python function `zscore(values)` that returns a list of population z-scores (mean 0, population standard deviation 1).",
     "import math\n"
     "r = zscore([2,4,4,4,5,5,7,9])\n"
     "assert abs(sum(r)) < 1e-9\n"
     "assert abs(r[0] - (-1.5)) < 1e-6\n"
     "assert abs(r[-1] - 2.0) < 1e-6",
     "def zscore(values):\n"
     "    n=len(values); m=sum(values)/n\n"
     "    s=(sum((v-m)**2 for v in values)/n)**0.5\n"
     "    return [(v-m)/s for v in values]",
     "def zscore(values):\n"
     "    n=len(values); m=sum(values)/n\n"
     "    s=(sum((v-m)**2 for v in values)/(n-1))**0.5\n"  # 標本標準偏差の誤り
     "    return [(v-m)/s for v in values]")
code("pearson_code_001", "stats_impl",
     "Write a Python function `pearson(xs, ys)` returning the Pearson correlation coefficient.",
     "assert abs(pearson([1,2,3,4],[2,4,6,8]) - 1.0) < 1e-6\n"
     "assert abs(pearson([1,2,3,4],[8,6,4,2]) + 1.0) < 1e-6\n"
     "assert abs(pearson([1,2,3],[1,2,3]) - 1.0) < 1e-6",
     "def pearson(xs, ys):\n"
     "    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n\n"
     "    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))\n"
     "    dx=(sum((x-mx)**2 for x in xs))**0.5\n"
     "    dy=(sum((y-my)**2 for y in ys))**0.5\n"
     "    return num/(dx*dy)",
     "def pearson(xs, ys):\n"
     "    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n\n"
     "    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/n")  # 共分散を返す誤り
code("samp_var_code_001", "stats_impl",
     "Write a Python function `sample_variance(values)` using the n-1 (Bessel) denominator.",
     "assert abs(sample_variance([2,4,4,4,5,5,7,9]) - 32/7) < 1e-6\n"
     "assert abs(sample_variance([1,2,3]) - 1.0) < 1e-9",
     "def sample_variance(values):\n"
     "    n=len(values); m=sum(values)/n\n"
     "    return sum((v-m)**2 for v in values)/(n-1)",
     "def sample_variance(values):\n"
     "    n=len(values); m=sum(values)/n\n"
     "    return sum((v-m)**2 for v in values)/n")  # 母分散の誤り
code("median_code_001", "stats_impl",
     "Write a Python function `median(values)` that returns the median (average of the two middle values for even length).",
     "assert median([3,1,2]) == 2\n"
     "assert median([4,1,3,2]) == 2.5\n"
     "assert median([5]) == 5",
     "def median(values):\n"
     "    s=sorted(values); n=len(s); mid=n//2\n"
     "    return s[mid] if n%2 else (s[mid-1]+s[mid])/2",
     "def median(values):\n"
     "    s=sorted(values); return s[len(s)//2]")  # 偶数長で誤り
code("accuracy_code_001", "stats_impl",
     "Write a Python function `accuracy(y_true, y_pred)` returning the classification accuracy (fraction of matching pairs).",
     "assert abs(accuracy([1,0,1,1],[1,0,0,1]) - 0.75) < 1e-9\n"
     "assert accuracy([1,1],[0,0]) == 0.0\n"
     "assert accuracy([2,2,2],[2,2,2]) == 1.0",
     "def accuracy(y_true, y_pred):\n"
     "    c=sum(1 for a,b in zip(y_true,y_pred) if a==b)\n"
     "    return c/len(y_true)",
     "def accuracy(y_true, y_pred):\n"
     "    return sum(1 for a,b in zip(y_true,y_pred) if a==b)")  # 割り算忘れ


# ===========================================================================
# scale_consistency: 桁・スケールの一貫性
# ===========================================================================
num("scale_001", "scale_consistency",
    "What is 12.34 multiplied by 10? Answer with the numeric value only.",
    12.34*10, 1e-4,
    ["Multiply the number twelve point three four by ten. Number only.",
     "Compute 12.34 * 10 and return just the value.",
     "Shift 12.34 one decimal place (times ten). Value only."])
num("scale_002", "scale_consistency",
    "Round 3.14159 to 2 decimal places. Answer with the numeric value only.",
    round(3.14159, 2), 1e-4,
    ["Round pi (3.14159) to two decimals. Number only.",
     "What is 3.14159 with two digits after the decimal point? Value only."])
num("scale_003", "scale_consistency",
    "Express 1/8 as a decimal. Answer with the numeric value only.",
    1/8, 1e-4,
    ["What is one eighth written as a decimal number? Value only.",
     "Divide 1 by 8 and give the decimal result only."])
num("scale_004", "scale_consistency",
    "What is 0.1 + 0.2, rounded to 1 decimal place? Answer with the numeric value only.",
    round(0.1+0.2, 1), 1e-4,
    ["Add one tenth and two tenths and round to one decimal. Number only.",
     "Compute 0.1 plus 0.2 (one decimal). Value only."])
num("scale_005", "scale_consistency",
    "What is 12.34 multiplied by 100? Answer with the numeric value only.",
    12.34*100, 1e-4,
    ["Multiply 12.34 by one hundred. Number only.",
     "Compute 12.34 * 100. Value only."])
num("scale_006", "scale_consistency",
    "What is 12.34 divided by 10? Answer with the numeric value only.",
    12.34/10, 1e-4,
    ["Divide twelve point three four by ten. Number only.",
     "Compute 12.34 / 10. Value only."])
num("scale_007", "scale_consistency",
    "Convert 25% to a decimal fraction. Answer with the numeric value only.",
    25/100, 1e-4,
    ["What is twenty-five percent expressed as a decimal? Number only.",
     "25 percent as a plain decimal fraction. Value only."])
num("scale_008", "scale_consistency",
    "Convert the decimal 0.75 to a percentage (just the number, no percent sign). Answer with the numeric value only.",
    0.75*100, 1e-4,
    ["Express 0.75 as a percentage value (number only, omit the % sign).",
     "What percent is 0.75? Give just the number."])
num("scale_009", "scale_consistency",
    "Evaluate 1.5e3 (scientific notation) as an ordinary decimal number. Answer with the numeric value only.",
    1.5e3, 1e-4,
    ["What is 1.5 x 10^3 written as a normal number? Value only.",
     "Convert 1.5e3 to plain decimal notation. Number only."])
num("scale_010", "scale_consistency",
    "Convert 2500 grams to kilograms. Answer with the numeric value only.",
    2500/1000, 1e-4,
    ["How many kilograms is 2500 grams? Number only.",
     "Divide 2500 grams by 1000 to get kilograms. Value only."])
num("scale_011", "scale_consistency",
    "Express 7/8 as a decimal. Answer with the numeric value only.",
    7/8, 1e-4,
    ["What is seven eighths as a decimal? Value only.",
     "Divide 7 by 8. Number only."])
num("scale_012", "scale_consistency",
    "Round 12345 to 2 significant figures. Answer with the numeric value only.",
    12000, 1e-4,
    ["What is 12345 rounded to two significant figures? Number only.",
     "Keep only the first two significant digits of 12345 (rounding). Value only."])
num("scale_013", "scale_consistency",
    "Round 7.896 to 2 decimal places. Answer with the numeric value only.",
    round(7.896, 2), 1e-4,
    ["What is 7.896 to two decimal places? Number only.",
     "Round the number 7.896 to the hundredths place. Value only."])
num("scale_014", "scale_consistency",
    "What is 3 divided by 4, as a decimal? Answer with the numeric value only.",
    3/4, 1e-4,
    ["Express the fraction three quarters as a decimal. Number only.",
     "Compute 3/4. Value only."])

code("round_code_001", "scale_consistency",
     "Write a Python function `to_kg(grams)` converting grams to kilograms.",
     "assert abs(to_kg(2500) - 2.5) < 1e-9\n"
     "assert abs(to_kg(0) - 0.0) < 1e-9\n"
     "assert abs(to_kg(750) - 0.75) < 1e-9",
     "def to_kg(grams):\n    return grams/1000",
     "def to_kg(grams):\n    return grams*1000")
code("percent_code_001", "scale_consistency",
     "Write a Python function `to_percent(frac)` converting a decimal fraction to a percentage number (0.25 -> 25.0).",
     "assert abs(to_percent(0.25) - 25.0) < 1e-9\n"
     "assert abs(to_percent(1.0) - 100.0) < 1e-9\n"
     "assert abs(to_percent(0.005) - 0.5) < 1e-9",
     "def to_percent(frac):\n    return frac*100",
     "def to_percent(frac):\n    return frac/100")
code("sigfig_code_001", "scale_consistency",
     "Write a Python function `round_sig(x, n)` rounding x to n significant figures.",
     "assert round_sig(12345, 2) == 12000\n"
     "assert round_sig(0.023456, 3) == 0.0235\n"
     "assert round_sig(9.99, 1) == 10.0",
     "import math\n"
     "def round_sig(x, n):\n"
     "    if x == 0:\n        return 0.0\n"
     "    d = math.ceil(math.log10(abs(x)))\n"
     "    return round(x, -int(d - n))",
     "def round_sig(x, n):\n    return round(x, n)")  # 小数点以下桁数と混同した誤り
code("frac_code_001", "scale_consistency",
     "Write a Python function `frac_to_decimal(numer, denom)` returning the decimal value of a fraction.",
     "assert abs(frac_to_decimal(1, 8) - 0.125) < 1e-9\n"
     "assert abs(frac_to_decimal(7, 8) - 0.875) < 1e-9\n"
     "assert abs(frac_to_decimal(3, 4) - 0.75) < 1e-9",
     "def frac_to_decimal(numer, denom):\n    return numer/denom",
     "def frac_to_decimal(numer, denom):\n    return numer//denom")  # 整数除算の誤り


# ===========================================================================
# discrete_continuous: 離散(フラグ/カテゴリ)と連続(実数)の混在処理
# ===========================================================================
num("disc_cont_001", "discrete_continuous",
    "Given values [1, 2, 3, 4] and boolean flags [True, False, True, False], compute the sum of the values whose flag is True. Answer with the numeric value only.",
    sum(v for v, f in zip([1, 2, 3, 4], [True, False, True, False]) if f), 1e-4,
    ["From the list 1,2,3,4 keep only entries marked True by [True,False,True,False] and add them. Number only.",
     "Sum the flagged elements: values 1,2,3,4 with mask True,False,True,False. Value only."])
num("disc_cont_002", "discrete_continuous",
    "Given values [10, 20, 30, 40, 50], count how many are strictly greater than 25. Answer with the numeric value only.",
    sum(1 for v in [10, 20, 30, 40, 50] if v > 25), 1e-4,
    ["How many of 10,20,30,40,50 exceed 25 (strictly)? Number only.",
     "Count entries above the threshold 25 in [10,20,30,40,50]. Value only."])
num("disc_cont_003", "discrete_continuous",
    "Compute the weighted mean of values [10, 20, 30] with weights [1, 2, 3]. Answer with the numeric value only.",
    sum(v*w for v, w in zip([10, 20, 30], [1, 2, 3]))/sum([1, 2, 3]), 0.01,
    ["Weighted average of 10,20,30 using weights 1,2,3 (weights sum to 6). Number only.",
     "(10*1 + 20*2 + 30*3) / (1+2+3) = ? Value only."])
num("disc_cont_004", "discrete_continuous",
    "Given labels ['a','b','a','b'] and values [1.0, 10.0, 3.0, 20.0], compute the mean of the values labelled 'a'. Answer with the numeric value only.",
    statistics.fmean([1.0, 3.0]), 1e-4,
    ["Average only the values whose label is 'a', given labels a,b,a,b and values 1,10,3,20. Number only.",
     "Group by label; report the mean of group 'a' for values [1,10,3,20]. Value only."])
num("disc_cont_005", "discrete_continuous",
    "Given flags [True, False, True, True, False], what fraction are True? Answer with the numeric value only.",
    sum([True, False, True, True, False])/5, 1e-4,
    ["What proportion of [True,False,True,True,False] are True? Number only.",
     "Divide the count of True by 5 for the mask True,False,True,True,False. Value only."])
num("disc_cont_006", "discrete_continuous",
    "Given values [5, 8, 12, 3] and boolean flags [False, True, True, False], compute the mean of the flagged (True) values. Answer with the numeric value only.",
    statistics.fmean([8, 12]), 1e-4,
    ["Average the values whose flag is True: values 5,8,12,3 with mask False,True,True,False. Number only.",
     "Mean of the flagged entries in [5,8,12,3] under [False,True,True,False]. Value only."])

code("disc_cont_code_001", "discrete_continuous",
     "Write a Python function `mean_of_flagged(values, flags)` that returns the arithmetic mean of the values whose corresponding flag is True. Return 0.0 if none are flagged.",
     "assert abs(mean_of_flagged([1,2,3,4], [True,False,True,False]) - 2.0) < 1e-9\n"
     "assert abs(mean_of_flagged([10,20,30], [False,False,True]) - 30.0) < 1e-9\n"
     "assert mean_of_flagged([1,2], [False,False]) == 0.0",
     "def mean_of_flagged(values, flags):\n"
     "    sel=[v for v,f in zip(values,flags) if f]\n"
     "    return sum(sel)/len(sel) if sel else 0.0",
     "def mean_of_flagged(values, flags):\n"
     "    sel=[v for v,f in zip(values,flags) if f]\n"
     "    return sum(sel)/len(sel)")  # 空で ZeroDivisionError
code("disc_cont_code_002", "discrete_continuous",
     "Write a Python function `category_means(labels, values)` that returns a dict mapping each category label to the mean of its values.",
     "r = category_means(['a','b','a','b'], [1.0, 10.0, 3.0, 20.0])\n"
     "assert abs(r['a'] - 2.0) < 1e-9\n"
     "assert abs(r['b'] - 15.0) < 1e-9",
     "def category_means(labels, values):\n"
     "    from collections import defaultdict\n"
     "    g=defaultdict(list)\n"
     "    for l,v in zip(labels,values): g[l].append(v)\n"
     "    return {k: sum(vs)/len(vs) for k,vs in g.items()}",
     "def category_means(labels, values):\n"
     "    return {l: v for l,v in zip(labels,values)}")  # 上書きで平均でない
code("count_above_code_001", "discrete_continuous",
     "Write a Python function `count_above(values, threshold)` returning how many values are strictly greater than threshold.",
     "assert count_above([10,20,30,40,50], 25) == 3\n"
     "assert count_above([1,2,3], 3) == 0\n"
     "assert count_above([5,5,5], 4) == 3",
     "def count_above(values, threshold):\n"
     "    return sum(1 for v in values if v > threshold)",
     "def count_above(values, threshold):\n"
     "    return sum(1 for v in values if v >= threshold)")  # >= の誤り
code("weighted_mean_code_001", "discrete_continuous",
     "Write a Python function `weighted_mean(values, weights)` returning the weighted arithmetic mean.",
     "assert abs(weighted_mean([10,20,30],[1,2,3]) - 23.3333) < 1e-3\n"
     "assert abs(weighted_mean([1,1,1],[5,5,5]) - 1.0) < 1e-9\n"
     "assert abs(weighted_mean([2,4],[1,1]) - 3.0) < 1e-9",
     "def weighted_mean(values, weights):\n"
     "    return sum(v*w for v,w in zip(values,weights))/sum(weights)",
     "def weighted_mean(values, weights):\n"
     "    return sum(v*w for v,w in zip(values,weights))/len(values)")  # 重み和でなく件数
code("group_sum_code_001", "discrete_continuous",
     "Write a Python function `group_sum(labels, values)` returning a dict mapping each label to the sum of its values.",
     "r = group_sum(['x','y','x'], [1,2,3])\n"
     "assert r['x'] == 4 and r['y'] == 2\n"
     "r2 = group_sum(['a'], [10])\n"
     "assert r2['a'] == 10",
     "def group_sum(labels, values):\n"
     "    from collections import defaultdict\n"
     "    g=defaultdict(float)\n"
     "    for l,v in zip(labels,values): g[l]+=v\n"
     "    return dict(g)",
     "def group_sum(labels, values):\n"
     "    return {l:v for l,v in zip(labels,values)}")  # 上書きで和でない
code("indicator_code_001", "discrete_continuous",
     "Write a Python function `indicator(values, threshold)` returning a list of 1/0 flags for values strictly above threshold.",
     "assert indicator([1,5,3,8], 3) == [0,1,0,1]\n"
     "assert indicator([3,3,3], 3) == [0,0,0]\n"
     "assert indicator([], 0) == []",
     "def indicator(values, threshold):\n"
     "    return [1 if v > threshold else 0 for v in values]",
     "def indicator(values, threshold):\n"
     "    return [1 if v >= threshold else 0 for v in values]")  # >= の誤り


# ===========================================================================
# 検証
# ===========================================================================
def verify():
    errors = []

    # --- code: ref で pass, wrong で fail することを確認 ---
    for pid, ref, wrong, test in _CODE_CHECKS:
        if not check_correctness(ref, test):
            errors.append(f"[CODE ref FAIL] {pid}: 参照実装が test を通らない")
        if check_correctness(wrong, test):
            errors.append(f"[CODE weak] {pid}: 誤実装が test を通る (テストが弱い)")

    # --- numeric: parse_number の対象は「最後の数値」。expected を bare number で
    #     出したときに rel_tol 内で自己一致するか (自明だが桁溢れ等の番人) ---
    for p in PROBLEMS:
        if p["kind"] != "numeric":
            continue
        exp = p["expected"]
        s = repr(exp)
        import re
        m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace(",", ""))
        got = float(m[-1])
        denom = abs(exp) if exp != 0 else 1.0
        if abs(got - exp)/denom > p["rel_tol"]:
            errors.append(f"[NUM self] {pid}: {got} vs {exp}")

    return errors


def write_and_oracle_check(path):
    with open(path, "w", encoding="utf-8") as f:
        for p in PROBLEMS:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    probs = load_numeric_problems(path)

    def oracle(prompt, n):
        # prompt から該当問題を引く
        for p in PROBLEMS:
            if p["prompt"] == prompt or prompt in p.get("paraphrases", []):
                if p["kind"] == "numeric":
                    return [f"The answer is {p['expected']}"]
                # code: そのままだと ref を持っていないので id で引く
                for pid, ref, wrong, test in _CODE_CHECKS:
                    if pid == p["id"]:
                        return [ref]
        return [""]

    ncs = numeric_consistency_score(probs, oracle)
    pis = path_independence_score(probs, oracle)
    return probs, ncs, pis


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="書き出さず検証のみ")
    args = ap.parse_args()

    errs = verify()
    if errs:
        print("=== 検証エラー ===")
        for e in errs:
            print(" ", e)
        sys.exit(1)

    out = os.path.join(_ROOT, "data", "numeric_stats_eval", "problems.jsonl")
    if args.check:
        import tempfile
        out = os.path.join(tempfile.mkdtemp(), "problems.jsonl")
    probs, ncs, pis = write_and_oracle_check(out)

    from collections import Counter
    cats = Counter(p.category for p in probs)
    kinds = Counter(p.kind for p in probs)
    n_para = sum(1 for p in probs if p.paraphrases)

    print(f"総問題数: {len(probs)}")
    print(f"カテゴリ別: {dict(cats)}")
    print(f"種別: {dict(kinds)}")
    print(f"paraphrases 付き: {n_para}")
    print(f"\n[oracle] numeric_consistency: overall={ncs['numeric/overall']:.4f}")
    for k in sorted(ncs):
        if k != "numeric/overall":
            print(f"          {k}={ncs[k]:.4f}")
    print(f"[oracle] path_indep agreement={pis['path_indep/agreement']:.4f} "
          f"dispersion={pis['path_indep/dispersion']:.4f} "
          f"n_problems={pis['path_indep/n_problems']:.0f}")

    ok = (abs(ncs["numeric/overall"] - 1.0) < 1e-9
          and abs(pis["path_indep/agreement"] - 1.0) < 1e-9
          and pis["path_indep/dispersion"] < 1e-9)
    print("\n" + ("✓ 全検証パス (期待値・test すべてオラクルで整合)" if ok
                  else "✗ オラクル不整合あり"))
    sys.exit(0 if ok else 1)
