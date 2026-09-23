"""Bounded professional-solver checks for typed linear arithmetic.

The input is data, not solver code: coefficients are exact rationals and the
grammar contains no expressions, quantifiers, functions, or Python evaluation.
"""
from __future__ import annotations

import re
from fractions import Fraction
from itertools import product
from math import gcd

from researchclaw.pipeline.evidence_store import content_hash


class FormalProofError(ValueError):
    pass


_RATIONAL = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")
_RELATIONS = {"<", "<=", "==", "!=", ">=", ">"}
MAX_VARIABLES = 16
MAX_PREMISES = 64
MAX_BITS = 512
MAX_ENUM_POINTS = 10000


def _fraction(value, label):
    if type(value) is int:
        result = Fraction(value)
    elif isinstance(value, str) and _RATIONAL.fullmatch(value):
        result = Fraction(value)
    else:
        raise FormalProofError(f"{label} must be an integer or canonical rational string")
    if abs(result.numerator).bit_length() > MAX_BITS or result.denominator.bit_length() > MAX_BITS:
        raise FormalProofError(f"{label} exceeds the exact arithmetic bound")
    return result


def _constraint(value, variables, label):
    if (not isinstance(value, dict) or set(value) != {"coefficients", "relation", "constant"}
            or value["relation"] not in _RELATIONS or not isinstance(value["coefficients"], dict)
            or not value["coefficients"]):
        raise FormalProofError(f"Invalid {label}")
    if any(not isinstance(name, str) or name not in variables for name in value["coefficients"]):
        raise FormalProofError(f"Unknown variable in {label}")
    coefficients = {name: _fraction(coefficient, f"{label} coefficient")
                    for name, coefficient in value["coefficients"].items()}
    coefficients = {name: coefficient for name, coefficient in coefficients.items() if coefficient}
    if not coefficients:
        raise FormalProofError(f"{label} must contain a nonzero coefficient")
    return {"coefficients": coefficients, "relation": value["relation"],
            "constant": _fraction(value["constant"], f"{label} constant")}


def validate_linear_statement(statement):
    fields = {"kind", "domain", "variables", "premises", "conclusion"}
    if (not isinstance(statement, dict) or not fields <= set(statement)
            or set(statement) - fields - {"portable_certificate"}
            or statement.get("kind") != "linear_arithmetic"):
        raise FormalProofError("Invalid linear arithmetic statement")
    if statement["domain"] not in {"integer", "real"}:
        raise FormalProofError("Linear arithmetic domain must be integer or real")
    variables = statement["variables"]
    if (not isinstance(variables, list) or not 1 <= len(variables) <= MAX_VARIABLES
            or len(set(variables)) != len(variables)
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name) for name in variables)):
        raise FormalProofError("Linear arithmetic needs unique Python-style variables")
    premises = statement["premises"]
    if not isinstance(premises, list) or len(premises) > MAX_PREMISES:
        raise FormalProofError("Linear arithmetic exceeds the premise bound")
    normalized = {"domain": statement["domain"], "variables": variables,
                  "premises": [_constraint(item, variables, f"premise {index}")
                               for index, item in enumerate(premises, start=1)],
                  "conclusion": _constraint(statement["conclusion"], variables, "conclusion")}
    return normalized


def _statement_hash(statement):
    base = dict(statement)
    base.pop("portable_certificate", None)
    return content_hash(base)


def _relation(left, relation, right):
    return {"<": left < right, "<=": left <= right, "==": left == right,
            "!=": left != right, ">=": left >= right, ">": left > right}[relation]


def _negate(relation):
    return {"<": ">=", "<=": ">", "==": "!=", "!=": "==", ">=": "<", ">": "<="}[relation]


def _evaluate(constraint, values):
    left = sum((coefficient * values[name] for name, coefficient in constraint["coefficients"].items()), Fraction())
    return bool(_relation(left, constraint["relation"], constraint["constant"]))


def _render_fraction(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _closed_inequalities(normalized):
    """Expand non-strict premises to a canonical A*x <= b sequence."""
    rows = []
    for index, item in enumerate(normalized["premises"], start=1):
        relation = item["relation"]
        if relation not in {"<=", ">=", "=="}:
            return None
        if relation in {"<=", "=="}:
            rows.append((f"premise_{index}:upper", item["coefficients"], item["constant"]))
        if relation in {">=", "=="}:
            rows.append((f"premise_{index}:lower",
                         {name: -coefficient for name, coefficient in item["coefficients"].items()},
                         -item["constant"]))
    return rows


def _alternative_rows(normalized):
    """Split premises into closed rows (A*x <= b) and strict rows (C*x < d).

    Motzkin's theorem of alternatives applies to conjunctions of closed and
    strict linear inequalities. ``!=`` premises are skipped: they exclude
    points but assert no linear inequality, so they contribute no row — a
    contradiction carried by the remaining rows still refutes the system.
    """
    closed, strict = [], []
    for index, item in enumerate(normalized["premises"], start=1):
        relation = item["relation"]
        if relation == "!=":
            continue
        if relation in {"<=", "=="}:
            closed.append((f"premise_{index}:upper", item["coefficients"], item["constant"]))
        if relation in {">=", "=="}:
            closed.append((f"premise_{index}:lower",
                           {name: -coefficient for name, coefficient in item["coefficients"].items()},
                           -item["constant"]))
        elif relation == ">":
            strict.append((f"premise_{index}:strict",
                           {name: -coefficient for name, coefficient in item["coefficients"].items()},
                           -item["constant"]))
        elif relation == "<":
            strict.append((f"premise_{index}:strict", item["coefficients"], item["constant"]))
    return closed, strict


def _effective_equalities(normalized):
    """Derive the linear equalities that every solution satisfies exactly.

    Equality premises hold with equality directly, and a pair of premises
    pinning the same closed form from both sides (``C*x <= d`` together with
    ``C*x >= d``) forces ``C*x == d``. Strict and ``!=`` premises never
    contribute: they exclude points but assert no equation. ``!=`` premises
    do not invalidate the derivation — the equalities below hold at every
    remaining solution, so a congruence contradiction over them is sound.
    """
    equalities = []
    consumed = set()
    for index, item in enumerate(normalized["premises"], start=1):
        if item["relation"] == "==":
            equalities.append(([f"premise_{index}:upper", f"premise_{index}:lower"],
                               item["coefficients"], item["constant"]))
            consumed.add(index)
    for i, item in enumerate(normalized["premises"], start=1):
        if item["relation"] != "<=" or i in consumed:
            continue
        left = {name: value for name, value in item["coefficients"].items() if value != 0}
        for j, other in enumerate(normalized["premises"], start=1):
            if j == i or other["relation"] != ">=" or j in consumed:
                continue
            right = {name: value for name, value in other["coefficients"].items() if value != 0}
            if right == left and other["constant"] == item["constant"]:
                equalities.append(([f"premise_{i}:upper", f"premise_{j}:lower"],
                                   item["coefficients"], item["constant"]))
                consumed.update({i, j})
                break
    return equalities


def _conclusion_targets(normalized):
    item = normalized["conclusion"]
    relation = item["relation"]
    if relation not in {"<=", ">=", "=="}:
        return None
    targets = []
    if relation in {"<=", "=="}:
        targets.append(("upper", item["coefficients"], item["constant"]))
    if relation in {">=", "=="}:
        targets.append(("lower", {name: -coefficient for name, coefficient in item["coefficients"].items()},
                        -item["constant"]))
    return targets


def verify_farkas_certificate(statement, certificate):
    """Verify a portable exact-rational certificate without importing Z3."""
    normalized = validate_linear_statement(statement)
    integer_strengthening = normalized["domain"] == "integer"
    rows, targets = _closed_inequalities(normalized), _conclusion_targets(normalized)
    if rows is None or targets is None:
        raise FormalProofError("Farkas certificates require non-strict inequalities or equalities")
    fields = {"schema_version", "kind", "statement_hash", "premise_witness", "premise_rows", "claims"}
    if integer_strengthening:
        fields.add("proof_domain")
    expected_schema = 2 if integer_strengthening else 1
    expected_kind = "farkas_linear_implication_over_reals" if integer_strengthening else "farkas_linear_implication"
    if (not isinstance(certificate, dict) or set(certificate) != fields
            or certificate["schema_version"] != expected_schema or certificate["kind"] != expected_kind
            or (integer_strengthening and certificate["proof_domain"] != "real_superset_of_integer_domain")
            or certificate["statement_hash"] != _statement_hash(statement)
            or certificate["premise_rows"] != [row[0] for row in rows]):
        raise FormalProofError("Invalid Farkas certificate identity")
    witness = certificate["premise_witness"]
    if not isinstance(witness, dict) or set(witness) != set(normalized["variables"]):
        raise FormalProofError("Invalid premise consistency witness")
    witness = {name: _fraction(value, "premise witness") for name, value in witness.items()}
    if not all(_evaluate(item, witness) for item in normalized["premises"]):
        raise FormalProofError("Premise consistency witness is invalid")
    claims = certificate["claims"]
    if not isinstance(claims, list) or len(claims) != len(targets):
        raise FormalProofError("Farkas certificate has incomplete claims")
    for claim, (target_name, target_coefficients, target_constant) in zip(claims, targets):
        if not isinstance(claim, dict) or set(claim) != {"target", "multipliers", "combined_constant"} \
                or claim["target"] != target_name or not isinstance(claim["multipliers"], list) \
                or len(claim["multipliers"]) != len(rows):
            raise FormalProofError("Invalid Farkas certificate claim")
        multipliers = [_fraction(value, "Farkas multiplier") for value in claim["multipliers"]]
        if any(value < 0 for value in multipliers):
            raise FormalProofError("Farkas multipliers must be nonnegative")
        combined = {name: sum((multiplier * coefficients.get(name, Fraction())
                               for multiplier, (_, coefficients, _) in zip(multipliers, rows)), Fraction())
                    for name in normalized["variables"]}
        wanted = {name: target_coefficients.get(name, Fraction()) for name in normalized["variables"]}
        constant = sum((multiplier * bound for multiplier, (_, _, bound) in zip(multipliers, rows)), Fraction())
        if combined != wanted or constant > target_constant \
                or _fraction(claim["combined_constant"], "combined constant") != constant:
            raise FormalProofError("Farkas linear combination does not prove its target")
    return True


def verify_inconsistency_certificate(statement, certificate):
    """Verify an exact-rational proof that the closed premises are unsatisfiable.

    Farkas' alternative: an infeasible closed system admits nonnegative
    multipliers combining the rows to ``0 <= negative constant``. For the
    integer domain this proves unsatisfiability of the real relaxation,
    which is a superset of the integer points (same argument as the v2
    implication certificates). Strict and ``!=`` premises are skipped: the
    multipliers may only combine the closed rows, so a valid certificate
    refutes a subset of the premises — which still refutes the conjunction.
    """
    normalized = validate_linear_statement(statement)
    integer_domain = normalized["domain"] == "integer"
    rows = _alternative_rows(normalized)[0]
    fields = {"schema_version", "kind", "statement_hash", "premise_rows",
              "multipliers", "combined_constant"}
    if integer_domain:
        fields.add("proof_domain")
    expected_schema = 2 if integer_domain else 1
    expected_kind = ("farkas_linear_inconsistency_over_reals" if integer_domain
                     else "farkas_linear_inconsistency")
    if (not isinstance(certificate, dict) or set(certificate) != fields
            or certificate["schema_version"] != expected_schema or certificate["kind"] != expected_kind
            or (integer_domain and certificate["proof_domain"] != "real_superset_of_integer_domain")
            or certificate["statement_hash"] != _statement_hash(statement)
            or certificate["premise_rows"] != [row[0] for row in rows]):
        raise FormalProofError("Invalid inconsistency certificate identity")
    multipliers = [_fraction(value, "inconsistency multiplier")
                   for value in certificate["multipliers"]]
    if len(multipliers) != len(rows) or any(value < 0 for value in multipliers):
        raise FormalProofError("Inconsistency multipliers must be nonnegative and complete")
    combined = {name: sum((multiplier * coefficients.get(name, Fraction())
                           for multiplier, (_, coefficients, _) in zip(multipliers, rows)), Fraction())
                for name in normalized["variables"]}
    constant = sum((multiplier * bound for multiplier, (_, _, bound) in zip(multipliers, rows)), Fraction())
    if any(value != 0 for value in combined.values()) or constant >= 0 \
            or _fraction(certificate["combined_constant"], "combined constant") != constant:
        raise FormalProofError("Inconsistency multipliers do not derive a contradiction")
    return True


def verify_motzkin_inconsistency_certificate(statement, certificate):
    """Verify an exact-rational proof that strict+closed premises are unsatisfiable.

    Motzkin's theorem of alternatives: the conjunction of closed rows
    ``A*x <= b`` and strict rows ``C*x < d`` is infeasible iff there are
    ``lambda >= 0`` and ``mu >= 0`` with ``mu != 0`` such that
    ``A^T*lambda + C^T*mu = 0`` and ``b^T*lambda + d^T*mu <= 0``. The
    nonzero-mu condition is what strictness costs against the plain Farkas
    alternative: without it, any feasible system could be "refuted" by the
    all-zero strict multipliers. For the integer domain this proves the
    real relaxation (a superset of the integer points) inconsistent.
    """
    normalized = validate_linear_statement(statement)
    integer_domain = normalized["domain"] == "integer"
    split = _alternative_rows(normalized)
    fields = {"schema_version", "kind", "statement_hash", "premise_rows", "strict_premise_rows",
              "closed_multipliers", "strict_multipliers", "combined_constant"}
    if integer_domain:
        fields.add("proof_domain")
    expected_schema = 2 if integer_domain else 1
    expected_kind = ("motzkin_linear_inconsistency_over_reals" if integer_domain
                     else "motzkin_linear_inconsistency")
    if (split is None
            or not isinstance(certificate, dict) or set(certificate) != fields
            or certificate["schema_version"] != expected_schema or certificate["kind"] != expected_kind
            or (integer_domain and certificate["proof_domain"] != "real_superset_of_integer_domain")
            or certificate["statement_hash"] != _statement_hash(statement)
            or certificate["premise_rows"] != [row[0] for row in split[0]]
            or certificate["strict_premise_rows"] != [row[0] for row in split[1]]):
        raise FormalProofError("Invalid Motzkin certificate identity")
    closed_multipliers = [_fraction(value, "Motzkin closed multiplier")
                          for value in certificate["closed_multipliers"]]
    strict_multipliers = [_fraction(value, "Motzkin strict multiplier")
                          for value in certificate["strict_multipliers"]]
    if (len(closed_multipliers) != len(split[0]) or any(value < 0 for value in closed_multipliers)
            or len(strict_multipliers) != len(split[1]) or any(value < 0 for value in strict_multipliers)
            or not any(value > 0 for value in strict_multipliers)):
        raise FormalProofError("Motzkin multipliers must be nonnegative with a positive strict multiplier")
    rows = split[0] + split[1]
    multipliers = closed_multipliers + strict_multipliers
    combined = {name: sum((multiplier * coefficients.get(name, Fraction())
                           for multiplier, (_, coefficients, _) in zip(multipliers, rows)), Fraction())
                for name in normalized["variables"]}
    constant = sum((multiplier * bound for multiplier, (_, _, bound) in zip(multipliers, rows)), Fraction())
    if any(value != 0 for value in combined.values()) or constant > 0 \
            or _fraction(certificate["combined_constant"], "combined constant") != constant:
        raise FormalProofError("Motzkin multipliers do not derive a contradiction")
    return True


def verify_counterexample_witness(statement, certificate):
    """Verify an exact-rational point that satisfies the premises and violates the conclusion."""
    normalized = validate_linear_statement(statement)
    fields = {"schema_version", "kind", "statement_hash", "witness"}
    if (not isinstance(certificate, dict) or set(certificate) != fields
            or certificate["schema_version"] != 1
            or certificate["kind"] != "linear_counterexample_witness"
            or certificate["statement_hash"] != _statement_hash(statement)):
        raise FormalProofError("Invalid counterexample certificate identity")
    witness = certificate["witness"]
    if not isinstance(witness, dict) or set(witness) != set(normalized["variables"]):
        raise FormalProofError("Invalid counterexample witness shape")
    values = {name: _fraction(value, "counterexample witness") for name, value in witness.items()}
    if normalized["domain"] == "integer" and any(value.denominator != 1 for value in values.values()):
        raise FormalProofError("Integer counterexample witness must be integral")
    if not all(_evaluate(item, values) for item in normalized["premises"]) \
            or _evaluate(normalized["conclusion"], values):
        raise FormalProofError("Counterexample witness does not satisfy premises and violate the conclusion")
    return True


def verify_bounded_integer_infeasibility(statement, certificate):
    """Verify an exact-rational proof that the premises have no integer point.

    The certificate first proves real-relaxation bounds for every variable:
    for the upper direction of ``v`` the multipliers combine the closed and
    strict premise rows into ``+e_v`` with combined constant ``c`` (giving
    ``v <= c``), and for the lower direction into ``-e_v`` (giving
    ``v >= -c``). The bound is strict exactly when some strict multiplier is
    positive — a strict row is strictly satisfied at any feasible point and
    positive multipliers preserve strictness — so the verifier derives the
    strictness from the multipliers instead of trusting a flag, then applies
    the exact integer rounding rules and enumerates the resulting integer
    box, requiring zero premise-satisfying points. Unbounded directions are
    rejected: no proven box, no certificate.
    """
    normalized = validate_linear_statement(statement)
    if normalized["domain"] != "integer":
        raise FormalProofError("Bounded integer infeasibility requires the integer domain")
    split = _alternative_rows(normalized)
    fields = {"schema_version", "kind", "statement_hash", "premise_rows", "strict_premise_rows",
              "bounds", "bound_proofs", "checked_assignments"}
    if (split is None
            or not isinstance(certificate, dict) or set(certificate) != fields
            or certificate["schema_version"] != 1
            or certificate["kind"] != "bounded_integer_infeasibility"
            or certificate["statement_hash"] != _statement_hash(statement)
            or certificate["premise_rows"] != [row[0] for row in split[0]]
            or certificate["strict_premise_rows"] != [row[0] for row in split[1]]):
        raise FormalProofError("Invalid bounded integer infeasibility certificate identity")
    closed_rows, strict_rows = split
    rows = closed_rows + strict_rows
    bounds = certificate["bounds"]
    proofs = certificate["bound_proofs"]
    if (not isinstance(bounds, dict) or set(bounds) != set(normalized["variables"])
            or not isinstance(proofs, dict) or set(proofs) != set(normalized["variables"])):
        raise FormalProofError("Invalid bounded integer infeasibility box shape")
    checked = certificate["checked_assignments"]
    if type(checked) is not int or checked < 0:
        raise FormalProofError("Invalid checked assignment count")
    integer_bounds = {}
    for name in normalized["variables"]:
        for direction in ("lower", "upper"):
            proof = proofs[name].get(direction)
            if not isinstance(proof, dict) or set(proof) != {"closed_multipliers",
                                                             "strict_multipliers", "constant"}:
                raise FormalProofError(f"Missing {direction} bound proof for {name}")
            closed_multipliers = [_fraction(value, "bound multiplier")
                                  for value in proof["closed_multipliers"]]
            strict_multipliers = [_fraction(value, "bound multiplier")
                                  for value in proof["strict_multipliers"]]
            if len(closed_multipliers) != len(closed_rows) or any(value < 0 for value in closed_multipliers) \
                    or len(strict_multipliers) != len(strict_rows) or any(value < 0 for value in strict_multipliers):
                raise FormalProofError("Bound multipliers must be nonnegative and complete")
            multipliers = closed_multipliers + strict_multipliers
            combined = {variable: sum((multiplier * coefficients.get(variable, Fraction())
                                       for multiplier, (_, coefficients, _) in zip(multipliers, rows)),
                                      Fraction())
                        for variable in normalized["variables"]}
            sign = 1 if direction == "upper" else -1
            wanted = {variable: sign * Fraction(1) if variable == name else Fraction()
                      for variable in normalized["variables"]}
            constant = sum((multiplier * bound for multiplier, (_, _, bound) in zip(multipliers, rows)),
                           Fraction())
            if combined != wanted or _fraction(proof["constant"], "bound constant") != constant:
                raise FormalProofError("Bound multipliers do not prove their declared bound")
            strict = any(value > 0 for value in strict_multipliers)
            real_bound = constant if direction == "upper" else -constant
            if direction == "lower":
                integer_bounds[name, "lower"] = (real_bound.numerator // real_bound.denominator + 1
                                                 if strict
                                                 else -((-real_bound.numerator) // real_bound.denominator))
            else:
                integer_bounds[name, "upper"] = (real_bound.numerator // real_bound.denominator
                                                 if not strict
                                                 else -((-real_bound.numerator) // real_bound.denominator) - 1)
        declared = bounds[name]
        if (not isinstance(declared, dict) or set(declared) != {"lower", "upper"}
                or type(declared["lower"]) is not int or type(declared["upper"]) is not int
                or declared["lower"] != integer_bounds[name, "lower"]
                or declared["upper"] != integer_bounds[name, "upper"]):
            raise FormalProofError("Declared integer bounds differ from proven bounds")
    total = 1
    for name in normalized["variables"]:
        lower, upper = integer_bounds[name, "lower"], integer_bounds[name, "upper"]
        if lower > upper:
            total = 0
            break
        total *= upper - lower + 1
        if total > MAX_ENUM_POINTS:
            raise FormalProofError("Bounded integer infeasibility box exceeds the enumeration limit")
    if checked != total:
        raise FormalProofError("Checked assignment count differs from the replayed box size")
    if total == 0:
        return True
    names = normalized["variables"]
    for point in product(*[range(integer_bounds[name, "lower"], integer_bounds[name, "upper"] + 1)
                           for name in names]):
        values = {name: Fraction(value) for name, value in zip(names, point)}
        if all(_evaluate(item, values) for item in normalized["premises"]):
            raise FormalProofError("Bounded integer infeasibility box contains a satisfying point")
    return True


def _congruence_scale_and_modulus(combined):
    """Integer scaling of a derived row and the gcd modulus of its coefficients."""
    scale = 1
    for value in combined.values():
        scale = scale * value.denominator // gcd(scale, value.denominator)
    modulus = 0
    for value in combined.values():
        modulus = gcd(modulus, abs((value * scale).numerator))
    return scale, modulus


def verify_integer_congruence_infeasibility(statement, certificate):
    """Verify an exact-rational proof that the premises have no integer point.

    Any rational combination of the equalities every solution satisfies
    (equality premises and opposite closed pairs) is itself an implied
    equality ``L*x == c``. Scaling the coefficients to integers by ``S``
    makes the left side a multiple of ``g = gcd(|L'_i|)`` at every integer
    point, so ``S*c`` must be divisible by ``g``; a combination whose scaled
    constant fails this divisibility excludes every integer solution. The
    argument needs integrality, so real-domain statements are rejected.
    """
    normalized = validate_linear_statement(statement)
    if normalized["domain"] != "integer":
        raise FormalProofError("Integer congruence infeasibility requires the integer domain")
    equalities = _effective_equalities(normalized)
    fields = {"schema_version", "kind", "statement_hash", "equality_premise_rows",
              "row_multipliers", "derived_coefficients", "derived_constant", "modulus"}
    if (not isinstance(certificate, dict) or set(certificate) != fields
            or certificate["schema_version"] != 1
            or certificate["kind"] != "integer_congruence_infeasibility"
            or certificate["statement_hash"] != _statement_hash(statement)
            or certificate["equality_premise_rows"] != [labels for labels, _, _ in equalities]):
        raise FormalProofError("Invalid integer congruence infeasibility certificate identity")
    multipliers = certificate["row_multipliers"]
    if not isinstance(multipliers, list) or len(multipliers) != len(equalities):
        raise FormalProofError("Row multipliers must match the effective equalities")
    row_multipliers = [_fraction(value, "congruence multiplier") for value in multipliers]
    combined = {name: sum((multiplier * coefficients.get(name, Fraction())
                           for multiplier, (_, coefficients, _) in zip(row_multipliers, equalities)),
                          Fraction())
                for name in normalized["variables"]}
    constant = sum((multiplier * bound for multiplier, (_, _, bound) in zip(row_multipliers, equalities)),
                   Fraction())
    declared = certificate["derived_coefficients"]
    if (not isinstance(declared, dict) or set(declared) != set(normalized["variables"])
            or {name: _fraction(value, "derived coefficient") for name, value in declared.items()} != combined
            or _fraction(certificate["derived_constant"], "derived constant") != constant):
        raise FormalProofError("Derived congruence row differs from the multiplier combination")
    if not any(value != 0 for value in combined.values()):
        if constant != 0:
            raise FormalProofError("Identically zero derived rows must be proven by a real "
                                   "infeasibility certificate")
        raise FormalProofError("Zero multiplier combinations derive no congruence")
    scale, modulus = _congruence_scale_and_modulus(combined)
    if type(certificate["modulus"]) is not int or certificate["modulus"] != modulus:
        raise FormalProofError("Declared modulus differs from the derived congruence")
    scaled_constant = constant * scale
    if scaled_constant.denominator != 1 or scaled_constant.numerator % modulus:
        return True
    raise FormalProofError("Derived congruence is satisfied by integer solutions; no contradiction")


def _integer_bounds(normalized):
    """Derive exact finite bounds from single-variable integer premises."""
    bounds = {name: [None, None] for name in normalized["variables"]}
    for item in normalized["premises"]:
        if len(item["coefficients"]) != 1 or item["relation"] == "!=":
            continue
        name, coefficient = next(iter(item["coefficients"].items()))
        relation, threshold = item["relation"], item["constant"] / coefficient
        if coefficient < 0:
            relation = {"<": ">", "<=": ">=", "==": "==", ">=": "<=", ">": "<"}[relation]
        lower = upper = None
        floor = threshold.numerator // threshold.denominator
        ceil = -((-threshold.numerator) // threshold.denominator)
        if relation == "<":
            upper = ceil - 1
        elif relation == "<=":
            upper = floor
        elif relation == ">=":
            lower = ceil
        elif relation == ">":
            lower = floor + 1
        elif threshold.denominator == 1:
            lower = upper = threshold.numerator
        else:
            return None
        if lower is not None:
            bounds[name][0] = lower if bounds[name][0] is None else max(bounds[name][0], lower)
        if upper is not None:
            bounds[name][1] = upper if bounds[name][1] is None else min(bounds[name][1], upper)
    if any(lower is None or upper is None or lower > upper for lower, upper in bounds.values()):
        return None
    points = 1
    for lower, upper in bounds.values():
        points *= upper - lower + 1
        if points > MAX_ENUM_POINTS:
            return None
    return bounds, points


def _enumerate_integer_domain(normalized, bounds):
    names = normalized["variables"]
    ranges = [range(bounds[name][0], bounds[name][1] + 1) for name in names]
    checked = satisfying = 0
    witness = None
    for point in product(*ranges):
        checked += 1
        values = {name: Fraction(value) for name, value in zip(names, point)}
        if all(_evaluate(item, values) for item in normalized["premises"]):
            satisfying += 1
            witness = witness or {name: int(values[name]) for name in names}
            if not _evaluate(normalized["conclusion"], values):
                raise FormalProofError("Exhaustive integer certificate has a counterexample")
    if witness is None:
        raise FormalProofError("Exhaustive integer certificate rejects vacuous implications")
    return checked, satisfying, witness


def verify_exhaustive_integer_certificate(statement, certificate):
    """Replay a finite QF_LIA domain exactly without importing Z3."""
    normalized = validate_linear_statement(statement)
    fields = {"schema_version", "kind", "statement_hash", "variables", "bounds",
              "checked_assignments", "satisfying_assignments", "premise_witness"}
    if (normalized["domain"] != "integer" or not isinstance(certificate, dict)
            or set(certificate) != fields or certificate["schema_version"] != 1
            or certificate["kind"] != "exhaustive_bounded_integer_implication"
            or certificate["statement_hash"] != _statement_hash(statement)
            or certificate["variables"] != normalized["variables"]):
        raise FormalProofError("Invalid exhaustive integer certificate identity")
    derived = _integer_bounds(normalized)
    if derived is None:
        raise FormalProofError("Statement has no bounded enumerable integer domain")
    bounds, points = derived
    rendered = {name: {"lower": lower, "upper": upper} for name, (lower, upper) in bounds.items()}
    if certificate["bounds"] != rendered or type(certificate["checked_assignments"]) is not int \
            or certificate["checked_assignments"] != points \
            or type(certificate["satisfying_assignments"]) is not int:
        raise FormalProofError("Exhaustive integer certificate bounds or counts differ")
    checked, satisfying, witness = _enumerate_integer_domain(normalized, bounds)
    if (checked != certificate["checked_assignments"] or satisfying != certificate["satisfying_assignments"]
            or certificate["premise_witness"] != witness):
        raise FormalProofError("Exhaustive integer certificate replay differs")
    return True


def _exhaustive_integer_certificate(statement, normalized):
    derived = _integer_bounds(normalized)
    if derived is None:
        return None
    bounds, _ = derived
    try:
        checked, satisfying, witness = _enumerate_integer_domain(normalized, bounds)
    except FormalProofError:
        return None
    certificate = {"schema_version": 1, "kind": "exhaustive_bounded_integer_implication",
                   "statement_hash": _statement_hash(statement), "variables": normalized["variables"],
                   "bounds": {name: {"lower": lower, "upper": upper}
                              for name, (lower, upper) in bounds.items()},
                   "checked_assignments": checked, "satisfying_assignments": satisfying,
                   "premise_witness": witness}
    verify_exhaustive_integer_certificate(statement, certificate)
    return certificate


def _z3_fraction(z3, value):
    if z3.is_int_value(value):
        return Fraction(value.as_long())
    if z3.is_rational_value(value):
        return Fraction(value.numerator_as_long(), value.denominator_as_long())
    raise FormalProofError("Z3 returned a non-rational value for linear arithmetic")


def _bounded_integer_infeasibility_certificate(statement, normalized, split, solver, exact, z3):
    """Search multiplier-proven real bounds and an exhaustive integer box.

    Runs after the Farkas and Motzkin inconsistency searches failed on an
    integer-domain statement: the real relaxation may still be feasible
    while every integer point is excluded (2x == 1, or x < 1 with 2x > 1).
    Each variable needs row-space bounds in both directions; systems whose
    rows only pin variables jointly (2x + 2y == 1) or not at all stay
    honestly certificate-less.
    """
    closed_rows, strict_rows = split
    rows = closed_rows + strict_rows

    def direction_proof(target):
        closed_mults = [z3.Real(f"bound_closed_{index}") for index in range(len(closed_rows))]
        strict_mults = [z3.Real(f"bound_strict_{index}") for index in range(len(strict_rows))]
        certificate_solver = solver()
        certificate_solver.add(*(value >= 0 for value in closed_mults + strict_mults))
        for name in normalized["variables"]:
            certificate_solver.add(
                sum((value * exact(coefficients.get(name, Fraction()))
                     for value, (_, coefficients, _) in zip(closed_mults, closed_rows)),
                    exact(Fraction()))
                + sum((value * exact(coefficients.get(name, Fraction()))
                       for value, (_, coefficients, _) in zip(strict_mults, strict_rows)),
                      exact(Fraction()))
                == exact(target.get(name, Fraction())))
        if certificate_solver.check() != z3.sat:
            return None
        model = certificate_solver.model()
        exact_closed = [_z3_fraction(z3, model.eval(value, model_completion=True))
                        for value in closed_mults]
        exact_strict = [_z3_fraction(z3, model.eval(value, model_completion=True))
                        for value in strict_mults]
        constant = sum((value * bound for value, (_, _, bound) in zip(exact_closed, closed_rows)), Fraction()) \
            + sum((value * bound for value, (_, _, bound) in zip(exact_strict, strict_rows)), Fraction())
        return {"closed_multipliers": [_render_fraction(value) for value in exact_closed],
                "strict_multipliers": [_render_fraction(value) for value in exact_strict],
                "constant": _render_fraction(constant)}, constant, any(value > 0 for value in exact_strict)

    bounds = {}
    proofs = {}
    for name in normalized["variables"]:
        lower_proof = direction_proof({name: Fraction(-1)})
        upper_proof = direction_proof({name: Fraction(1)})
        if lower_proof is None or upper_proof is None:
            return None
        proofs[name] = {"lower": lower_proof[0], "upper": upper_proof[0]}
        lower_real = -lower_proof[1]
        upper_real = upper_proof[1]
        lower = lower_real.numerator // lower_real.denominator + 1 if lower_proof[2] \
            else -((-lower_real.numerator) // lower_real.denominator)
        upper = upper_real.numerator // upper_real.denominator if not upper_proof[2] \
            else -((-upper_real.numerator) // upper_real.denominator) - 1
        bounds[name] = {"lower": lower, "upper": upper}
    total = 1
    for name in normalized["variables"]:
        lower, upper = bounds[name]["lower"], bounds[name]["upper"]
        if lower > upper:
            total = 0
            break
        total *= upper - lower + 1
        if total > MAX_ENUM_POINTS:
            return None
    if total:
        names = normalized["variables"]
        for point in product(*[range(bounds[name]["lower"], bounds[name]["upper"] + 1) for name in names]):
            values = {name: Fraction(value) for name, value in zip(names, point)}
            if all(_evaluate(item, values) for item in normalized["premises"]):
                raise FormalProofError("Z3 integer unsat contradicted by a satisfying bounded point")
    certificate = {"schema_version": 1, "kind": "bounded_integer_infeasibility",
                   "statement_hash": _statement_hash(statement),
                   "premise_rows": [row[0] for row in closed_rows],
                   "strict_premise_rows": [row[0] for row in strict_rows],
                   "bounds": bounds, "bound_proofs": proofs, "checked_assignments": total}
    verify_bounded_integer_infeasibility(statement, certificate)
    return certificate


def _primitive_integer_row(coefficients, constant):
    """Rescale a rational row to integer coefficients with content gcd one."""
    scale = 1
    for value in list(coefficients.values()) + [constant]:
        scale = scale * value.denominator // gcd(scale, value.denominator)
    scaled = {name: value * scale for name, value in coefficients.items()}
    target = constant * scale
    modulus = 0
    for value in scaled.values():
        modulus = gcd(modulus, abs(value.numerator))
    if modulus > 1:
        scaled = {name: value / modulus for name, value in scaled.items()}
        target = target / modulus
    return scaled, target


def _congruence_certificate_from_combo(statement, normalized, equalities, combo):
    """Freeze a multiplier combination into a verified congruence certificate."""
    multipliers = [combo.get(index, Fraction()) for index in range(len(equalities))]
    combined = {name: sum((multiplier * coefficients.get(name, Fraction())
                           for multiplier, (_, coefficients, _) in zip(multipliers, equalities)),
                          Fraction())
                for name in normalized["variables"]}
    constant = sum((multiplier * bound for multiplier, (_, _, bound) in zip(multipliers, equalities)),
                   Fraction())
    _, modulus = _congruence_scale_and_modulus(combined)
    certificate = {"schema_version": 1, "kind": "integer_congruence_infeasibility",
                   "statement_hash": _statement_hash(statement),
                   "equality_premise_rows": [labels for labels, _, _ in equalities],
                   "row_multipliers": [_render_fraction(value) for value in multipliers],
                   "derived_coefficients": {name: _render_fraction(value)
                                            for name, value in combined.items()},
                   "derived_constant": _render_fraction(constant),
                   "modulus": modulus}
    verify_integer_congruence_infeasibility(statement, certificate)
    return certificate


def _integer_congruence_infeasibility_certificate(statement, normalized):
    """Search a congruence obstruction among the implied equalities.

    Runs after the bounded-box search failed on an integer-domain statement:
    rows that pin variables only jointly (2x - 2y == 1) still carry modular
    obstructions. Echelon elimination combines the effective equalities with
    rational multipliers, testing each derived row in primitive form — a
    primitive row whose constant is not an integer excludes every integer
    point. Stored rows keep the exact multiplier combination so a frozen
    certificate reproduces the derived row it was found on. The
    elimination is sound but not complete: systems whose obstruction needs
    more than fraction-free elimination stay honestly certificate-less.
    """
    if normalized["domain"] != "integer":
        return None
    equalities = _effective_equalities(normalized)
    if not equalities:
        return None
    pool = []
    for index, (_, coefficients, constant) in enumerate(equalities):
        row = {"coefficients": dict(coefficients), "constant": constant,
               "combo": {index: Fraction(1)}, "pivoted": False}
        pool.append(row)
        scaled, target = _primitive_integer_row(coefficients, constant)
        if scaled and target.denominator != 1:
            return _congruence_certificate_from_combo(statement, normalized, equalities, row["combo"])
    owned = set()
    while True:
        pivot = variable = None
        for row in pool:
            if row["pivoted"]:
                continue
            for name in normalized["variables"]:
                if name not in owned and row["coefficients"].get(name, Fraction()) != 0:
                    pivot, variable = row, name
                    break
            if pivot is not None:
                break
        if pivot is None:
            return None
        pivot["pivoted"] = True
        owned.add(variable)
        coefficient = pivot["coefficients"][variable]
        for position, row in enumerate(pool):
            if row is pivot:
                continue
            factor = row["coefficients"].get(variable, Fraction())
            if factor == 0:
                continue
            combined_coefficients = {
                name: factor * pivot["coefficients"].get(name, Fraction())
                - coefficient * row["coefficients"].get(name, Fraction())
                for name in set(pivot["coefficients"]) | set(row["coefficients"])}
            combined_constant = factor * pivot["constant"] - coefficient * row["constant"]
            combined_combo = {index: factor * multiplier
                              for index, multiplier in pivot["combo"].items()}
            for index, multiplier in row["combo"].items():
                combined_combo[index] = combined_combo.get(index, Fraction()) - coefficient * multiplier
            combo = {index: multiplier for index, multiplier in combined_combo.items()
                     if multiplier != 0}
            pool[position] = {"coefficients": combined_coefficients,
                              "constant": combined_constant,
                              "combo": combo, "pivoted": False}
            scaled, target = _primitive_integer_row(combined_coefficients, combined_constant)
            if scaled and target.denominator != 1:
                return _congruence_certificate_from_combo(statement, normalized, equalities, combo)


def check_linear_arithmetic(statement):
    """Return the standard status/checker/evidence tuple.

    Backend absence and solver ``unknown`` are unresolved. Inconsistent
    premises are also unresolved, preventing vacuous implications from being
    presented as useful theorems.
    """
    normalized = validate_linear_statement(statement)
    checker = "z3_linear_arithmetic_implication/v2"
    if "portable_certificate" in statement:
        certificate = statement["portable_certificate"]
        kind = certificate.get("kind") if isinstance(certificate, dict) else None
        if kind == "exhaustive_bounded_integer_implication":
            verify_exhaustive_integer_certificate(statement, certificate)
            checker_name = "exact_bounded_integer_enumeration/v1"
            proof_domain = "bounded_integer"
        elif kind == "linear_counterexample_witness":
            verify_counterexample_witness(statement, certificate)
            witness = {name: _render_fraction(_fraction(value, "counterexample witness"))
                       for name, value in certificate["witness"].items()}
            return "disproved", "exact_fraction_counterexample/v1", {
                "statement_hash": _statement_hash(statement), "backend": "portable_certificate",
                "logic": "QF_LIA" if normalized["domain"] == "integer" else "QF_LRA",
                "implication_holds": False, "counterexample": witness,
                "counterexample_exactly_validated": True,
                "portable_certificate": certificate, "portable_certificate_verified": True}
        elif kind in ("farkas_linear_inconsistency", "farkas_linear_inconsistency_over_reals"):
            verify_inconsistency_certificate(statement, certificate)
            over_reals = kind == "farkas_linear_inconsistency_over_reals"
            return "unresolved", "exact_fraction_inconsistency/v1", {
                "statement_hash": _statement_hash(statement), "backend": "portable_certificate",
                "logic": "QF_LIA" if normalized["domain"] == "integer" else "QF_LRA",
                "premises_consistent": False, "vacuous_implication_rejected": True,
                "certificate_proof_domain": "real_superset_of_integer_domain" if over_reals else "real",
                "portable_certificate": certificate, "portable_certificate_verified": True}
        elif kind in ("motzkin_linear_inconsistency", "motzkin_linear_inconsistency_over_reals"):
            verify_motzkin_inconsistency_certificate(statement, certificate)
            over_reals = kind == "motzkin_linear_inconsistency_over_reals"
            return "unresolved", "exact_fraction_motzkin/v1", {
                "statement_hash": _statement_hash(statement), "backend": "portable_certificate",
                "logic": "QF_LIA" if normalized["domain"] == "integer" else "QF_LRA",
                "premises_consistent": False, "vacuous_implication_rejected": True,
                "certificate_proof_domain": "real_superset_of_integer_domain" if over_reals else "real",
                "portable_certificate": certificate, "portable_certificate_verified": True}
        elif kind == "bounded_integer_infeasibility":
            verify_bounded_integer_infeasibility(statement, certificate)
            return "unresolved", "exact_bounded_integer_infeasibility/v1", {
                "statement_hash": _statement_hash(statement), "backend": "portable_certificate",
                "logic": "QF_LIA",
                "premises_consistent": False, "vacuous_implication_rejected": True,
                "certificate_proof_domain": "integer",
                "portable_certificate": certificate, "portable_certificate_verified": True}
        elif kind == "integer_congruence_infeasibility":
            verify_integer_congruence_infeasibility(statement, certificate)
            return "unresolved", "exact_integer_congruence_infeasibility/v1", {
                "statement_hash": _statement_hash(statement), "backend": "portable_certificate",
                "logic": "QF_LIA",
                "premises_consistent": False, "vacuous_implication_rejected": True,
                "certificate_proof_domain": "integer",
                "portable_certificate": certificate, "portable_certificate_verified": True}
        else:
            verify_farkas_certificate(statement, certificate)
            checker_name = "exact_fraction_farkas/v2" if normalized["domain"] == "integer" else "exact_fraction_farkas/v1"
            proof_domain = "real"
        return "machine_checked", checker_name, {
            "statement_hash": _statement_hash(statement), "backend": "portable_certificate",
            "logic": "QF_LIA" if normalized["domain"] == "integer" else "QF_LRA",
            "certificate_proof_domain": proof_domain,
            "exact_rational_input": True,
            "portable_certificate": certificate, "portable_certificate_verified": True}
    evidence = {"statement_hash": _statement_hash(statement), "backend": "z3",
                "logic": "QF_LIA" if normalized["domain"] == "integer" else "QF_LRA",
                "exact_rational_input": True, "timeout_ms": 2000, "rlimit": 200000}
    try:
        import z3
    except ImportError:
        evidence["backend"] = "unavailable"
        return "unresolved", checker, evidence
    evidence["z3_version"] = z3.get_version_string()
    constructors = z3.Ints if normalized["domain"] == "integer" else z3.Reals
    symbols = dict(zip(normalized["variables"], constructors(" ".join(normalized["variables"]))))

    def exact(value):
        return z3.IntVal(value.numerator) if value.denominator == 1 else z3.RealVal(f"{value.numerator}/{value.denominator}")

    def encode(constraint, relation=None):
        left = sum((exact(coefficient) * symbols[name]
                    for name, coefficient in constraint["coefficients"].items()), exact(Fraction()))
        return _relation(left, relation or constraint["relation"], exact(constraint["constant"]))

    def solver():
        result = z3.SolverFor(evidence["logic"])
        result.set(timeout=evidence["timeout_ms"], rlimit=evidence["rlimit"])
        return result

    premises_solver = solver()
    premises_solver.add(*(encode(item) for item in normalized["premises"]))
    premises_result = premises_solver.check()
    evidence["premises_result"] = str(premises_result)
    if premises_result == z3.unknown:
        evidence["reason_unknown"] = premises_solver.reason_unknown()[:200]
        return "unresolved", checker, evidence
    if premises_result == z3.unsat:
        evidence.update({"premises_consistent": False, "vacuous_implication_rejected": True})
        split = _alternative_rows(normalized)

        def try_bounded_integer_infeasibility():
            if normalized["domain"] != "integer" or split is None \
                    or "portable_certificate" in evidence:
                return
            certificate = _bounded_integer_infeasibility_certificate(
                statement, normalized, split, solver, exact, z3)
            if certificate is not None:
                evidence.update({"portable_certificate": certificate,
                                 "portable_certificate_checker":
                                     "exact_bounded_integer_infeasibility/v1",
                                 "certificate_proof_domain": "integer",
                                 "portable_certificate_verified": True})

        def try_integer_congruence_infeasibility():
            if normalized["domain"] != "integer" or "portable_certificate" in evidence:
                return
            certificate = _integer_congruence_infeasibility_certificate(statement, normalized)
            if certificate is not None:
                evidence.update({"portable_certificate": certificate,
                                 "portable_certificate_checker":
                                     "exact_integer_congruence_infeasibility/v1",
                                 "certificate_proof_domain": "integer",
                                 "portable_certificate_verified": True})

        def try_farkas_inconsistency():
            if "portable_certificate" in evidence or split is None:
                return
            rows = split[0]
            # Farkas' alternative: search for nonnegative multipliers that
            # combine the closed rows into ``0 <= negative constant``. For
            # the integer domain this proves the real relaxation (a
            # superset of the integer points) inconsistent; an integer-only
            # contradiction like 2x == 1 has no such certificate and stays
            # honestly unresolved. Also reached from the Motzkin branch when
            # no positive strict multiplier works: a contradiction carried
            # by the closed rows alone still proves real infeasibility.
            multipliers = [z3.Real(f"inconsistency_{index}") for index in range(len(rows))]
            certificate_solver = solver()
            certificate_solver.add(*(value >= 0 for value in multipliers))
            for name in normalized["variables"]:
                certificate_solver.add(sum((value * exact(coefficients.get(name, Fraction()))
                                            for value, (_, coefficients, _) in zip(multipliers, rows)),
                                           exact(Fraction())) == exact(Fraction()))
            certificate_solver.add(sum((value * exact(bound)
                                        for value, (_, _, bound) in zip(multipliers, rows)),
                                       exact(Fraction())) < exact(Fraction()))
            if certificate_solver.check() == z3.sat:
                certificate_model = certificate_solver.model()
                exact_multipliers = [_z3_fraction(z3, certificate_model.eval(value, model_completion=True))
                                     for value in multipliers]
                combined_constant = sum((value * bound for value, (_, _, bound)
                                         in zip(exact_multipliers, rows)), Fraction())
                integer_domain = normalized["domain"] == "integer"
                certificate = {"schema_version": 2 if integer_domain else 1,
                               "kind": ("farkas_linear_inconsistency_over_reals" if integer_domain
                                        else "farkas_linear_inconsistency"),
                               "statement_hash": _statement_hash(statement),
                               "premise_rows": [row[0] for row in rows],
                               "multipliers": [_render_fraction(value) for value in exact_multipliers],
                               "combined_constant": _render_fraction(combined_constant)}
                if integer_domain:
                    certificate["proof_domain"] = "real_superset_of_integer_domain"
                verify_inconsistency_certificate(statement, certificate)
                evidence.update({"portable_certificate": certificate,
                                 "portable_certificate_checker": "exact_fraction_inconsistency/v1",
                                 "certificate_proof_domain": ("real_superset_of_integer_domain" if integer_domain
                                                              else "real"),
                                 "portable_certificate_verified": True})

        if split is not None and split[1]:
            # Motzkin's theorem of alternatives: search for nonnegative
            # multipliers over the closed and strict rows, with at least one
            # positive strict multiplier, that combine into ``0 <=
            # nonpositive constant``. For the integer domain this proves the
            # real relaxation (a superset of the integer points)
            # inconsistent; an integer-only contradiction that stays
            # real-feasible (x < 1 with 2x > 1) has no such certificate and
            # stays honestly unresolved.
            rows, strict_rows = split
            multipliers = [z3.Real(f"motzkin_closed_{index}") for index in range(len(rows))]
            strict_multipliers = [z3.Real(f"motzkin_strict_{index}") for index in range(len(strict_rows))]
            certificate_solver = solver()
            certificate_solver.add(*(value >= 0 for value in multipliers + strict_multipliers))
            certificate_solver.add(z3.Sum(strict_multipliers) > exact(Fraction()))
            for name in normalized["variables"]:
                certificate_solver.add(
                    sum((value * exact(coefficients.get(name, Fraction()))
                         for value, (_, coefficients, _) in zip(multipliers, rows)),
                        exact(Fraction()))
                    + sum((value * exact(coefficients.get(name, Fraction()))
                           for value, (_, coefficients, _) in zip(strict_multipliers, strict_rows)),
                          exact(Fraction()))
                    == exact(Fraction()))
            certificate_solver.add(
                sum((value * exact(bound) for value, (_, _, bound) in zip(multipliers, rows)),
                    exact(Fraction()))
                + sum((value * exact(bound) for value, (_, _, bound) in zip(strict_multipliers, strict_rows)),
                      exact(Fraction()))
                <= exact(Fraction()))
            if certificate_solver.check() == z3.sat:
                certificate_model = certificate_solver.model()
                exact_closed = [_z3_fraction(z3, certificate_model.eval(value, model_completion=True))
                                for value in multipliers]
                exact_strict = [_z3_fraction(z3, certificate_model.eval(value, model_completion=True))
                                for value in strict_multipliers]
                combined_constant = sum((value * bound for value, (_, _, bound)
                                         in zip(exact_closed, rows)), Fraction()) \
                    + sum((value * bound for value, (_, _, bound)
                           in zip(exact_strict, strict_rows)), Fraction())
                integer_domain = normalized["domain"] == "integer"
                certificate = {"schema_version": 2 if integer_domain else 1,
                               "kind": ("motzkin_linear_inconsistency_over_reals" if integer_domain
                                        else "motzkin_linear_inconsistency"),
                               "statement_hash": _statement_hash(statement),
                               "premise_rows": [row[0] for row in rows],
                               "strict_premise_rows": [row[0] for row in strict_rows],
                               "closed_multipliers": [_render_fraction(value) for value in exact_closed],
                               "strict_multipliers": [_render_fraction(value) for value in exact_strict],
                               "combined_constant": _render_fraction(combined_constant)}
                if integer_domain:
                    certificate["proof_domain"] = "real_superset_of_integer_domain"
                verify_motzkin_inconsistency_certificate(statement, certificate)
                evidence.update({"portable_certificate": certificate,
                                 "portable_certificate_checker": "exact_fraction_motzkin/v1",
                                 "certificate_proof_domain": ("real_superset_of_integer_domain" if integer_domain
                                                              else "real"),
                                 "portable_certificate_verified": True})
            try_farkas_inconsistency()
            try_bounded_integer_infeasibility()
            try_integer_congruence_infeasibility()
            return "unresolved", checker, evidence
        try_farkas_inconsistency()
        try_bounded_integer_infeasibility()
        try_integer_congruence_infeasibility()
        return "unresolved", checker, evidence
    evidence["premises_consistent"] = True
    premise_model = premises_solver.model()
    premise_witness = {name: _z3_fraction(z3, premise_model.eval(symbol, model_completion=True))
                       for name, symbol in symbols.items()}
    if not all(_evaluate(item, premise_witness) for item in normalized["premises"]):
        raise FormalProofError("Z3 premise witness failed independent exact validation")

    implication_solver = solver()
    implication_solver.add(*(encode(item) for item in normalized["premises"]))
    implication_solver.add(encode(normalized["conclusion"], _negate(normalized["conclusion"]["relation"])))
    result = implication_solver.check()
    evidence["negated_conclusion_result"] = str(result)
    if result == z3.unsat:
        evidence.update({"implication_holds": True, "proof_method": "unsatisfiable_negated_conclusion"})
        rows, targets = _closed_inequalities(normalized), _conclusion_targets(normalized)
        if rows is not None and targets is not None:
            claims = []
            for target_index, (target_name, target_coefficients, target_constant) in enumerate(targets):
                multipliers = [z3.Real(f"farkas_{target_index}_{index}") for index in range(len(rows))]
                certificate_solver = solver()
                certificate_solver.add(*(value >= 0 for value in multipliers))
                for name in normalized["variables"]:
                    certificate_solver.add(sum((value * exact(coefficients.get(name, Fraction()))
                                                for value, (_, coefficients, _) in zip(multipliers, rows)),
                                               exact(Fraction())) == exact(target_coefficients.get(name, Fraction())))
                certificate_solver.add(sum((value * exact(bound)
                                            for value, (_, _, bound) in zip(multipliers, rows)),
                                           exact(Fraction())) <= exact(target_constant))
                if certificate_solver.check() != z3.sat:
                    claims = []
                    break
                certificate_model = certificate_solver.model()
                exact_multipliers = [_z3_fraction(z3, certificate_model.eval(value, model_completion=True))
                                     for value in multipliers]
                combined_constant = sum((value * bound for value, (_, _, bound)
                                         in zip(exact_multipliers, rows)), Fraction())
                claims.append({"target": target_name,
                               "multipliers": [_render_fraction(value) for value in exact_multipliers],
                               "combined_constant": _render_fraction(combined_constant)})
            if claims:
                integer_strengthening = normalized["domain"] == "integer"
                certificate = {"schema_version": 2 if integer_strengthening else 1,
                               "kind": ("farkas_linear_implication_over_reals" if integer_strengthening
                                        else "farkas_linear_implication"),
                               "statement_hash": _statement_hash(statement),
                               "premise_witness": {name: _render_fraction(value)
                                                   for name, value in premise_witness.items()},
                               "premise_rows": [row[0] for row in rows], "claims": claims}
                if integer_strengthening:
                    certificate["proof_domain"] = "real_superset_of_integer_domain"
                verify_farkas_certificate(statement, certificate)
                evidence.update({"portable_certificate": certificate,
                                 "portable_certificate_checker": ("exact_fraction_farkas/v2" if integer_strengthening
                                                                  else "exact_fraction_farkas/v1"),
                                 "certificate_proof_domain": "real",
                                 "portable_certificate_verified": True})
            else:
                evidence["portable_certificate"] = None
        else:
            evidence["portable_certificate"] = None
        if normalized["domain"] == "integer" and evidence.get("portable_certificate") is None:
            certificate = _exhaustive_integer_certificate(statement, normalized)
            if certificate is not None:
                evidence.update({"portable_certificate": certificate,
                                 "portable_certificate_checker": "exact_bounded_integer_enumeration/v1",
                                 "certificate_proof_domain": "bounded_integer",
                                 "portable_certificate_verified": True})
        return "machine_checked", checker, evidence
    if result == z3.unknown:
        evidence["reason_unknown"] = implication_solver.reason_unknown()[:200]
        return "unresolved", checker, evidence

    model = implication_solver.model()
    values = {}
    rendered = {}
    for name, symbol in symbols.items():
        value = model.eval(symbol, model_completion=True)
        fraction = _z3_fraction(z3, value)
        values[name] = fraction
        rendered[name] = str(fraction)
    valid = (all(_evaluate(item, values) for item in normalized["premises"])
             and not _evaluate(normalized["conclusion"], values))
    if not valid:
        raise FormalProofError("Z3 counterexample failed independent exact validation")
    certificate = {"schema_version": 1, "kind": "linear_counterexample_witness",
                   "statement_hash": _statement_hash(statement),
                   "witness": {name: _render_fraction(value) for name, value in values.items()}}
    verify_counterexample_witness(statement, certificate)
    evidence.update({"implication_holds": False, "counterexample": rendered,
                     "counterexample_exactly_validated": True,
                     "portable_certificate": certificate,
                     "portable_certificate_checker": "exact_fraction_counterexample/v1",
                     "portable_certificate_verified": True})
    return "disproved", checker, evidence
