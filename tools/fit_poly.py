#!/usr/bin/env python3
"""Fit fixed-range polynomial approximations for the benchmark recurrence."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cpu_value(x, lane, niterations):
    value = float(x)
    for _ in range(niterations):
        if lane == 0:
            value += math.sqrt(math.log(value) + 1.0)
        elif lane == 1:
            value += math.sqrt(math.cos(value) + 1.0)
        elif lane == 2:
            value += math.sqrt(math.sin(value) + 1.0)
        else:
            value += math.sqrt(math.tan(value) + 1.0)
    return value


def solve_linear(a, b):
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        scale = a[col][col]
        for j in range(col, n):
            a[col][j] /= scale
        b[col] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, n):
                a[row][j] -= factor * a[col][j]
            b[row] -= factor * b[col]
    return b


def fit_least_squares(lane, degree, train_points, niterations):
    terms = degree + 1
    normal = [[0.0] * terms for _ in range(terms)]
    rhs = [0.0] * terms
    for i in range(train_points):
        x = 1.0 + 0.01 * i / (train_points - 1)
        s = (x - 1.005) * 200.0
        y = cpu_value(x, lane, niterations)
        powers = [1.0]
        for _ in range(1, terms):
            powers.append(powers[-1] * s)
        for row in range(terms):
            rhs[row] += powers[row] * y
            for col in range(terms):
                normal[row][col] += powers[row] * powers[col]
    return solve_linear(normal, rhs)


def eval_poly(coeffs, s):
    value = 0.0
    for coeff in reversed(coeffs):
        value = value * s + coeff
    return value


def validate(lane, coeffs, test_points, niterations):
    max_abs = 0.0
    max_rel = 0.0
    worst_x = 0.0
    for i in range(test_points):
        x = 1.0 + 0.01 * i / (test_points - 1)
        s = (x - 1.005) * 200.0
        actual = cpu_value(x, lane, niterations)
        approx = eval_poly(coeffs, s)
        abs_err = abs(actual - approx)
        rel_err = abs_err / abs(actual)
        if rel_err > max_rel:
            max_abs = abs_err
            max_rel = rel_err
            worst_x = x
    return {"max_abs": max_abs, "max_rel": max_rel, "worst_x": worst_x}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, default=4)
    parser.add_argument("--train-points", type=int, default=1001)
    parser.add_argument("--test-points", type=int, default=10001)
    parser.add_argument("--niterations", type=int, default=5)
    parser.add_argument("--rel-tol", type=float, default=1.0e-3)
    parser.add_argument("--output", default="reports/latest/poly_fits.json")
    args = parser.parse_args()

    results = []
    for lane in range(4):
        for degree in range(args.max_degree + 1):
            coeffs = fit_least_squares(
                lane, degree, args.train_points, args.niterations
            )
            metrics = validate(lane, coeffs, args.test_points, args.niterations)
            results.append(
                {
                    "lane": lane,
                    "degree": degree,
                    "passes": metrics["max_rel"] <= args.rel_tol,
                    "coefficients": coeffs,
                    **metrics,
                }
            )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")

    print("lane degree pass max_rel max_abs coefficients")
    for item in results:
        coeffs = ",".join(f"{coeff:.9g}f" for coeff in item["coefficients"])
        print(
            f"{item['lane']} {item['degree']} "
            f"{'yes' if item['passes'] else 'no'} "
            f"{item['max_rel']:.6g} {item['max_abs']:.6g} {coeffs}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
