"""Bounded professional-solver checks for typed linear arithmetic.

The input is data, not solver code: coefficients are exact rationals and the
grammar contains no expressions, quantifiers, functions, or Python evaluation.
"""
from __future__ import annotations

import re
from fractions import Fraction
from itertools import combinations, product
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


def _cg_scale_row(coefficients, constant, variables):
    """Scale a rational closed row to integers: ``(A, B)`` with ``A*x <= B``.

    The scale is the lcm of the denominators of every coefficient and the
    constant, so both sides become exact integers (``int`` is exact here,
    not truncating).
    """
    scale = 1
    for name in variables:
        value = coefficients.get(name, Fraction())
        scale = scale * value.denominator // gcd(scale, value.denominator)
    scale = scale * constant.denominator // gcd(scale, constant.denominator)
    return ([int(coefficients.get(name, Fraction()) * scale) for name in variables],
            int(constant * scale))


def _cg_rounded_cut(rows, multipliers):
    """Derive one Chvátal–Gomory cut from nonnegative integer multipliers.

    ``rows`` are the scaled premise rows ``(A, B, strict)``; the combination
    ``C*x < d`` (strict exactly when a strict row carries positive weight)
    is rescaled to a primitive integer coefficient vector of content ``g``
    and the constant rounds down to ``floor(d/g)`` — ``ceil(d/g) - 1`` when
    strict. Returns ``(A, B)`` for the closed cut row, or ``None`` when the
    combination has no coefficients (real-relaxation territory, never a
    cut).
    """
    width = len(rows[0][0])
    combined = [0 for _ in range(width)]
    constant = 0
    strict = False
    for multiplier, (coefficients, bound, row_strict) in zip(multipliers, rows):
        if multiplier:
            strict = strict or row_strict
            for index in range(width):
                combined[index] += multiplier * coefficients[index]
            constant += multiplier * bound
    modulus = 0
    for value in combined:
        modulus = gcd(modulus, abs(value))
    if modulus == 0:
        return None
    if strict:
        rounded = -((-constant) // modulus) - 1
    else:
        rounded = constant // modulus
    return ([value // modulus for value in combined], rounded)


def verify_integer_cg_cut_infeasibility(statement, certificate):
    """Verify an exact Chvátal–Gomory cut refutation of integer feasibility.

    Each cut combines closed and strict premise rows with nonnegative
    multipliers into ``C*x < d`` (strict exactly when a strict row carries
    positive weight), rescales to a primitive integer coefficient vector of
    content ``g`` and rounds the constant down to ``floor(d/g)`` — or
    ``ceil(d/g) - 1`` when strict. At every integer point the left side is
    integral, so the rounded inequality still holds. A Farkas/Motzkin
    contradiction over the cuts and the premise rows (nonnegative
    multipliers, zero coefficients, nonpositive constant, positive strict
    weight required when the constant vanishes) then excludes every integer
    point while leaving the real relaxation untouched. The verifier
    re-derives every cut from its declared multipliers, re-derives the
    rounded rows, and re-checks the final contradiction itself.
    """
    normalized = validate_linear_statement(statement)
    if normalized["domain"] != "integer":
        raise FormalProofError("CG cut certificates require the integer domain")
    split = _alternative_rows(normalized)
    fields = {"schema_version", "kind", "statement_hash", "closed_premise_rows",
              "strict_premise_rows", "cuts", "cut_multipliers", "closed_multipliers",
              "strict_multipliers", "combined_constant"}
    if (split is None or not isinstance(certificate, dict) or set(certificate) != fields
            or certificate["schema_version"] != 1
            or certificate["kind"] != "integer_cg_cut_infeasibility"
            or certificate["statement_hash"] != _statement_hash(statement)
            or certificate["closed_premise_rows"] != [row[0] for row in split[0]]
            or certificate["strict_premise_rows"] != [row[0] for row in split[1]]):
        raise FormalProofError("Invalid CG cut certificate identity")
    variables = normalized["variables"]
    closed_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in split[0]]
    strict_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in split[1]]
    cut_rows = []
    for index, declared in enumerate(certificate["cuts"]):
        label = f"cut {index}"
        if not isinstance(declared, dict) or set(declared) != {
                "closed_multipliers", "strict_multipliers",
                "derived_coefficients", "derived_constant"}:
            raise FormalProofError(f"Invalid {label} identity")
        closed_multipliers = [_fraction(value, f"{label} closed multiplier")
                              for value in declared["closed_multipliers"]]
        strict_multipliers = [_fraction(value, f"{label} strict multiplier")
                              for value in declared["strict_multipliers"]]
        if len(closed_multipliers) != len(closed_scaled) or any(value < 0 for value in closed_multipliers) \
                or len(strict_multipliers) != len(strict_scaled) or any(value < 0 for value in strict_multipliers):
            raise FormalProofError(f"{label} multipliers must be nonnegative and complete")
        if any(value.denominator != 1 for value in closed_multipliers + strict_multipliers):
            # Rational multipliers add nothing: clearing their common
            # denominator yields the same primitive cut after content
            # division, so cut derivations are pinned to integers.
            raise FormalProofError(f"{label} multipliers must be integers")
        if not any(value != 0 for value in closed_multipliers + strict_multipliers):
            raise FormalProofError(f"{label} multipliers derive no cut")
        cut = _cg_rounded_cut([(coefficients, bound, False) for coefficients, bound in closed_scaled]
                              + [(coefficients, bound, True) for coefficients, bound in strict_scaled],
                              [int(value) for value in closed_multipliers]
                              + [int(value) for value in strict_multipliers])
        if cut is None:
            raise FormalProofError(f"{label} combination has no coefficients")
        cut_rows.append(cut[:2])
        expected_coefficients = {name: _render_fraction(Fraction(cut[0][position]))
                                 for position, name in enumerate(variables)}
        if (declared["derived_coefficients"] != expected_coefficients
                or _fraction(declared["derived_constant"], f"{label} derived constant") != cut[1]):
            raise FormalProofError(f"Declared {label} differs from the multiplier combination")
    cut_multipliers = [_fraction(value, "cut witness multiplier")
                       for value in certificate["cut_multipliers"]]
    closed_multipliers = [_fraction(value, "closed witness multiplier")
                          for value in certificate["closed_multipliers"]]
    strict_multipliers = [_fraction(value, "strict witness multiplier")
                          for value in certificate["strict_multipliers"]]
    if len(cut_multipliers) != len(cut_rows) or any(value < 0 for value in cut_multipliers) \
            or len(closed_multipliers) != len(closed_scaled) or any(value < 0 for value in closed_multipliers) \
            or len(strict_multipliers) != len(strict_scaled) or any(value < 0 for value in strict_multipliers):
        raise FormalProofError("Witness multipliers must be nonnegative and complete")
    if not any(value != 0 for value in cut_multipliers):
        raise FormalProofError("CG contradiction must use at least one cut row")
    combined = {name: 0 for name in variables}
    constant = Fraction(0)
    for multiplier, (coefficients, bound) in zip(cut_multipliers, cut_rows):
        if multiplier:
            for position, name in enumerate(variables):
                combined[name] += multiplier * coefficients[position]
            constant += multiplier * bound
    for multiplier, (coefficients, bound) in zip(closed_multipliers, closed_scaled):
        if multiplier:
            for position, name in enumerate(variables):
                combined[name] += multiplier * coefficients[position]
            constant += multiplier * bound
    for multiplier, (coefficients, bound) in zip(strict_multipliers, strict_scaled):
        if multiplier:
            for position, name in enumerate(variables):
                combined[name] += multiplier * coefficients[position]
            constant += multiplier * bound
    if (any(value != 0 for value in combined.values())
            or _fraction(certificate["combined_constant"], "combined constant") != constant
            or constant > 0
            or (constant == 0 and not any(value != 0 for value in strict_multipliers))):
        raise FormalProofError("Witness multipliers do not derive a CG contradiction")
    return True


def verify_integer_iterated_cg_cut_infeasibility(statement, certificate):
    """Verify an exact two-round Chvátal–Gomory cut refutation.

    Round one re-derives every declared cut as a unit-multiplier
    Chvátal–Gomory cut of a single premise row (focused feedback stays one
    CG step deep). Round two re-derives every declared cut from the
    re-derived round-one rows and the premise rows with nonnegative
    integer multipliers over the merged index space — round-one rows
    first, then closed premises, then strict premises. The witness
    combines the round-two cuts, the round-one rows, and the premise rows
    with nonnegative multipliers into a contradiction with zero
    coefficients and a nonpositive constant, needing positive strict
    weight when the constant vanishes; at least one round-two cut must
    carry weight.
    """
    normalized = validate_linear_statement(statement)
    if normalized["domain"] != "integer":
        raise FormalProofError("CG cut certificates require the integer domain")
    split = _alternative_rows(normalized)
    fields = {"schema_version", "kind", "statement_hash", "closed_premise_rows",
              "strict_premise_rows", "round1_cuts", "cuts", "cut_multipliers",
              "round1_multipliers", "closed_multipliers", "strict_multipliers",
              "combined_constant"}
    if (split is None or not isinstance(certificate, dict) or set(certificate) != fields
            or certificate["schema_version"] != 1
            or certificate["kind"] != "integer_iterated_cg_cut_infeasibility"
            or certificate["statement_hash"] != _statement_hash(statement)
            or certificate["closed_premise_rows"] != [row[0] for row in split[0]]
            or certificate["strict_premise_rows"] != [row[0] for row in split[1]]):
        raise FormalProofError("Invalid iterated CG cut certificate identity")
    variables = normalized["variables"]
    closed_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in split[0]]
    strict_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in split[1]]
    premise_rows = ([(coefficients, bound, False) for coefficients, bound in closed_scaled]
                    + [(coefficients, bound, True) for coefficients, bound in strict_scaled])
    round1_rows = []
    for index, declared in enumerate(certificate["round1_cuts"]):
        label = f"round-1 cut {index}"
        if not isinstance(declared, dict) or set(declared) != {
                "closed_multipliers", "strict_multipliers",
                "derived_coefficients", "derived_constant"}:
            raise FormalProofError(f"Invalid {label} identity")
        closed_multipliers = [_fraction(value, f"{label} closed multiplier")
                              for value in declared["closed_multipliers"]]
        strict_multipliers = [_fraction(value, f"{label} strict multiplier")
                              for value in declared["strict_multipliers"]]
        if len(closed_multipliers) != len(closed_scaled) \
                or len(strict_multipliers) != len(strict_scaled) \
                or any(value < 0 for value in closed_multipliers + strict_multipliers) \
                or any(value.denominator != 1
                       for value in closed_multipliers + strict_multipliers):
            raise FormalProofError(f"{label} multipliers must be nonnegative integers")
        multipliers = [int(value) for value in closed_multipliers + strict_multipliers]
        if sum(multipliers) != 1:
            raise FormalProofError(f"{label} derivation must be one unit multiplier")
        cut = _cg_rounded_cut(premise_rows, multipliers)
        if cut is None:
            raise FormalProofError(f"{label} derivation yields no cut")
        expected_coefficients = {name: _render_fraction(Fraction(cut[0][position]))
                                 for position, name in enumerate(variables)}
        if (declared["derived_coefficients"] != expected_coefficients
                or _fraction(declared["derived_constant"], f"{label} derived constant") != cut[1]):
            raise FormalProofError(f"Declared {label} differs from the multiplier combination")
        round1_rows.append(cut[:2])
    merged_rows = ([(coefficients, bound, False) for coefficients, bound in round1_rows]
                   + [(coefficients, bound, False) for coefficients, bound in closed_scaled]
                   + [(coefficients, bound, True) for coefficients, bound in strict_scaled])
    merged_closed_count = len(round1_rows) + len(closed_scaled)
    cut_rows = []
    for index, declared in enumerate(certificate["cuts"]):
        label = f"cut {index}"
        if not isinstance(declared, dict) or set(declared) != {
                "closed_multipliers", "strict_multipliers",
                "derived_coefficients", "derived_constant"}:
            raise FormalProofError(f"Invalid {label} identity")
        closed_multipliers = [_fraction(value, f"{label} closed multiplier")
                              for value in declared["closed_multipliers"]]
        strict_multipliers = [_fraction(value, f"{label} strict multiplier")
                              for value in declared["strict_multipliers"]]
        if len(closed_multipliers) != merged_closed_count or any(value < 0 for value in closed_multipliers) \
                or len(strict_multipliers) != len(strict_scaled) or any(value < 0 for value in strict_multipliers):
            raise FormalProofError(f"{label} multipliers must be nonnegative and complete")
        if any(value.denominator != 1 for value in closed_multipliers + strict_multipliers):
            # Rational multipliers add nothing: clearing their common
            # denominator yields the same primitive cut after content
            # division, so cut derivations are pinned to integers.
            raise FormalProofError(f"{label} multipliers must be integers")
        if not any(value != 0 for value in closed_multipliers + strict_multipliers):
            raise FormalProofError(f"{label} multipliers derive no cut")
        cut = _cg_rounded_cut(merged_rows,
                              [int(value) for value in closed_multipliers]
                              + [int(value) for value in strict_multipliers])
        if cut is None:
            raise FormalProofError(f"{label} combination has no coefficients")
        cut_rows.append(cut[:2])
        expected_coefficients = {name: _render_fraction(Fraction(cut[0][position]))
                                 for position, name in enumerate(variables)}
        if (declared["derived_coefficients"] != expected_coefficients
                or _fraction(declared["derived_constant"], f"{label} derived constant") != cut[1]):
            raise FormalProofError(f"Declared {label} differs from the multiplier combination")
    cut_multipliers = [_fraction(value, "cut witness multiplier")
                       for value in certificate["cut_multipliers"]]
    round1_multipliers = [_fraction(value, "round-1 witness multiplier")
                          for value in certificate["round1_multipliers"]]
    closed_multipliers = [_fraction(value, "closed witness multiplier")
                          for value in certificate["closed_multipliers"]]
    strict_multipliers = [_fraction(value, "strict witness multiplier")
                          for value in certificate["strict_multipliers"]]
    if len(cut_multipliers) != len(cut_rows) or any(value < 0 for value in cut_multipliers) \
            or len(round1_multipliers) != len(round1_rows) or any(value < 0 for value in round1_multipliers) \
            or len(closed_multipliers) != len(closed_scaled) or any(value < 0 for value in closed_multipliers) \
            or len(strict_multipliers) != len(strict_scaled) or any(value < 0 for value in strict_multipliers):
        raise FormalProofError("Witness multipliers must be nonnegative and complete")
    if not any(value != 0 for value in cut_multipliers):
        raise FormalProofError("CG contradiction must use at least one round-two cut row")
    combined = {name: 0 for name in variables}
    constant = Fraction(0)
    for multiplier, (coefficients, bound) in zip(cut_multipliers, cut_rows):
        if multiplier:
            for position, name in enumerate(variables):
                combined[name] += multiplier * coefficients[position]
            constant += multiplier * bound
    for multiplier, (coefficients, bound) in zip(round1_multipliers, round1_rows):
        if multiplier:
            for position, name in enumerate(variables):
                combined[name] += multiplier * coefficients[position]
            constant += multiplier * bound
    for multiplier, (coefficients, bound) in zip(closed_multipliers, closed_scaled):
        if multiplier:
            for position, name in enumerate(variables):
                combined[name] += multiplier * coefficients[position]
            constant += multiplier * bound
    for multiplier, (coefficients, bound) in zip(strict_multipliers, strict_scaled):
        if multiplier:
            for position, name in enumerate(variables):
                combined[name] += multiplier * coefficients[position]
            constant += multiplier * bound
    if (any(value != 0 for value in combined.values())
            or _fraction(certificate["combined_constant"], "combined constant") != constant
            or constant > 0
            or (constant == 0 and not any(value != 0 for value in strict_multipliers))):
        raise FormalProofError("Witness multipliers do not derive a CG contradiction")
    return True


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
    obstructions. Two searches run in order. The echelon combines the
    effective equalities with rational multipliers, testing each derived row
    in primitive form — a primitive row whose constant is not an integer
    excludes every integer point; stored rows keep the exact multiplier
    combination so a frozen certificate reproduces the derived row it was
    found on. An obstruction can live on a non-basis combination of the row
    space, so when the echelon misses, the Smith-normal-form search runs and
    is complete for this equality-obstruction class (see its docstring).
    """
    if normalized["domain"] != "integer":
        return None
    equalities = _effective_equalities(normalized)
    if not equalities:
        return None
    certificate = _echelon_congruence_certificate(statement, normalized, equalities)
    if certificate is not None:
        return certificate
    return _snf_congruence_certificate(statement, normalized, equalities)


def _echelon_congruence_certificate(statement, normalized, equalities):
    """Fraction-free elimination search for a primitive-row congruence."""
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


def _snf_row_transform(matrix):
    """Unimodular row transform taking an integer matrix to diagonal form.

    Row operations are recorded in the returned transform (they carry the
    multiplier combinations the certificate stores); column operations only
    reshape the working copy. Each pivot pass either advances the pivot or
    strictly shrinks the smallest nonzero absolute entry of the trailing
    submatrix, so the loop terminates.
    """
    working = [list(row) for row in matrix]
    total_rows = len(working)
    width = len(working[0])
    transform = [[1 if i == j else 0 for j in range(total_rows)] for i in range(total_rows)]

    def swap_rows(i, j):
        working[i], working[j] = working[j], working[i]
        transform[i], transform[j] = transform[j], transform[i]

    def add_row(i, j, factor):
        if factor:
            working[i] = [a + factor * b for a, b in zip(working[i], working[j])]
            transform[i] = [a + factor * b for a, b in zip(transform[i], transform[j])]

    def negate_row(i):
        working[i] = [-a for a in working[i]]
        transform[i] = [-a for a in transform[i]]

    def swap_columns(i, j):
        for row in working:
            row[i], row[j] = row[j], row[i]

    def add_column(i, j, factor):
        if factor:
            for row in working:
                row[i] += factor * row[j]

    pivot = 0
    while pivot < min(total_rows, width):
        candidate = None
        for i in range(pivot, total_rows):
            for j in range(pivot, width):
                if working[i][j] and (candidate is None
                                      or abs(working[i][j]) < abs(candidate[2])):
                    candidate = (i, j, working[i][j])
        if candidate is None:
            break
        swap_rows(pivot, candidate[0])
        swap_columns(pivot, candidate[1])
        if working[pivot][pivot] < 0:
            negate_row(pivot)
        moved = False
        for i in range(pivot + 1, total_rows):
            if working[i][pivot]:
                factor = working[i][pivot] // working[pivot][pivot]
                add_row(i, pivot, -factor)
                if working[i][pivot]:
                    moved = True
        for j in range(pivot + 1, width):
            if working[pivot][j]:
                factor = working[pivot][j] // working[pivot][pivot]
                add_column(j, pivot, -factor)
                if working[pivot][j]:
                    moved = True
        if not moved:
            divisible = True
            for i in range(pivot + 1, total_rows):
                for j in range(pivot + 1, width):
                    if working[i][j] % working[pivot][pivot]:
                        divisible = False
                        add_row(pivot, i, 1)
                        break
                if not divisible:
                    break
            if divisible:
                pivot += 1
    return transform


def _snf_congruence_certificate(statement, normalized, equalities):
    """Smith-normal-form search: complete for the equality-obstruction class.

    The echelon only inspects basis rows of the equality row space, but the
    obstruction may need an arbitrary combination. With ``C`` the integer
    matrix of the lcm-scaled equality rows and ``U`` its unimodular row
    transform, any row of ``U`` combines the equalities exactly, and the
    implied equality ``L*x == c`` (``L = U_i*C``, ``c = U_i*d``) has no
    integer point exactly when ``g = gcd(|L|)`` does not divide ``c``.
    Completeness: ``U*C*V`` is diagonal with entries ``s_i``, so
    ``U_i*C = s_i*(V^-1)_i`` where ``V^-1`` is unimodular — every one of its
    rows is primitive, making ``gcd(U_i*C)`` exactly ``s_i``. An integer
    solution of ``C*x = d`` forces ``s_i | (U*d)_i`` on every row, so no row
    can qualify; when the equalities are real-feasible but have no integer
    point, the Smith divisibility condition fails at some pivot, and that
    row qualifies. All-zero combined rows with a nonzero constant mean the
    equalities alone are real-infeasible — Farkas territory — and stay
    certificate-less here.
    """
    variables = normalized["variables"]
    rows = []
    constants = []
    scales = []
    for _, coefficients, constant in equalities:
        scale = 1
        for value in list(coefficients.values()) + [constant]:
            scale = scale * value.denominator // gcd(scale, value.denominator)
        if all(coefficients.get(name, Fraction()) == 0 for name in variables):
            if constant != 0:
                return None
            continue
        scales.append(Fraction(scale))
        rows.append([int(coefficients.get(name, Fraction()) * scale) for name in variables])
        constants.append(int(constant * scale))
    if not rows:
        return None
    transform = _snf_row_transform(rows)
    for row in transform:
        combined = [sum(row[index] * rows[index][position] for index in range(len(rows)))
                    for position in range(len(variables))]
        constant = sum(row[index] * constants[index] for index in range(len(rows)))
        modulus = 0
        for value in combined:
            modulus = gcd(modulus, abs(value))
        if modulus == 0:
            if constant != 0:
                return None
            continue
        if constant % modulus:
            combo = {index: Fraction(row[index]) * scales[index]
                     for index in range(len(rows)) if row[index]}
            return _congruence_certificate_from_combo(statement, normalized, equalities, combo)
    return None


_CG_COMBINATION_ROW_LIMIT = 8
_CG_COMBINATION_TIERS = ((4, 2), (3, 3))
_CG_ITERATED_ROW_LIMIT = 16
_CG_WITNESS_RETRY_TIMEOUT_MS = 8000
_CG_WITNESS_RETRY_RLIMIT = 800000


def _cg_cut_pool(closed_scaled, strict_scaled):
    """Enumerate candidate Chvátal–Gomory cuts with their derivations.

    Rows are the scaled premise rows ``(A, B, strict)``. Single-row cuts
    (one unit multiplier per row) come first — the band class closes with
    those alone — followed by bounded combinations behind the staged
    tiers of :data:`_CG_COMBINATION_TIERS` (up to four rows with
    multipliers up to two, plus up to three rows with multipliers up to
    three), available only while the row count stays small. Entries
    deduplicate on the derived primitive cut, first derivation winning,
    so the pool and the search stay deterministic.
    """
    rows = [(coefficients, bound, False) for coefficients, bound in closed_scaled] \
        + [(coefficients, bound, True) for coefficients, bound in strict_scaled]
    closed_count = len(closed_scaled)
    candidates = [((index,), (1,)) for index in range(len(rows))]
    if len(rows) <= _CG_COMBINATION_ROW_LIMIT:
        for max_terms, max_multiplier in _CG_COMBINATION_TIERS:
            for size in range(2, min(max_terms, len(rows)) + 1):
                for combo in combinations(range(len(rows)), size):
                    candidates.extend((combo, multipliers)
                                      for multipliers
                                      in product(range(1, max_multiplier + 1),
                                                 repeat=size))
    pool = []
    seen = set()
    for indices, multipliers in candidates:
        closed_mults = [0] * closed_count
        strict_mults = [0] * (len(rows) - closed_count)
        for index, multiplier in zip(indices, multipliers):
            if index < closed_count:
                closed_mults[index] = multiplier
            else:
                strict_mults[index - closed_count] = multiplier
        cut = _cg_rounded_cut(rows, closed_mults + strict_mults)
        if cut is None:
            continue
        key = (tuple(cut[0]), cut[1])
        if key in seen:
            continue
        seen.add(key)
        pool.append((tuple(closed_mults), tuple(strict_mults), cut))
    return pool


def _cg_focused_cut_pool(merged_closed, strict_scaled, round1_count):
    """Candidate cuts for iteration over a merged space too wide for
    :func:`_cg_cut_pool`.

    The full staged enumeration is skipped once the merged row space
    exceeds :data:`_CG_COMBINATION_ROW_LIMIT`; this focused variant keeps
    the single-row cuts of every merged row but draws its bounded
    combinations only from the round-one cut rows (the first
    ``round1_count`` merged positions). Whenever the premise count itself
    stays within :data:`_CG_COMBINATION_ROW_LIMIT` — the regime every
    measured member lives in — combinations among the premise rows were
    already searched by the round-one pool, so the new expressive power
    comes from combining the fed-back cut rows; beyond that premise count,
    premise-by-premise combinations are unsearched in both rounds, an
    honest incompleteness on top of the one below. Combinations mixing a
    round-one row with a premise row are honestly not enumerated.
    Multiplier vectors index the full merged space, so the certificate
    format and the verifier are unchanged.
    """
    rows = [(coefficients, bound, False) for coefficients, bound in merged_closed] \
        + [(coefficients, bound, True) for coefficients, bound in strict_scaled]
    closed_count = len(merged_closed)
    cut_positions = list(range(round1_count))
    candidates = [((index,), (1,)) for index in range(len(rows))]
    for max_terms, max_multiplier in _CG_COMBINATION_TIERS:
        for size in range(2, min(max_terms, len(cut_positions)) + 1):
            for combo in combinations(cut_positions, size):
                candidates.extend((combo, multipliers)
                                  for multipliers
                                  in product(range(1, max_multiplier + 1),
                                             repeat=size))
    pool = []
    seen = set()
    for indices, multipliers in candidates:
        closed_mults = [0] * closed_count
        strict_mults = [0] * (len(rows) - closed_count)
        for index, multiplier in zip(indices, multipliers):
            if index < closed_count:
                closed_mults[index] = multiplier
            else:
                strict_mults[index - closed_count] = multiplier
        cut = _cg_rounded_cut(rows, closed_mults + strict_mults)
        if cut is None:
            continue
        key = (tuple(cut[0]), cut[1])
        if key in seen:
            continue
        seen.add(key)
        pool.append((tuple(closed_mults), tuple(strict_mults), cut))
    return pool


def _cg_cut_witness(pool, closed_scaled, strict_scaled, variables, solver, exact, z3):
    """Search a Farkas/Motzkin witness over ``pool`` cuts and premise rows.

    Nonnegative multipliers, zero combined coefficients, nonpositive
    constant, and either a negative constant or a positive strict weight;
    at least one cut must carry weight, since a witness over the premise
    rows alone is real infeasibility, which the Farkas layer already owns.
    The search runs a fast pass over the single-row cuts first — that LP
    has a handful of variables and closes the band class — and escalates
    to the full pool (combination cuts included) only when the fast pass
    finds no witness. An ``unsat`` answer is final; an ``unknown`` answer
    under the caller's resource budget (full pools over wide systems make
    the LP heavy) is retried once under a raised fixed budget before it
    counts as no witness. Returns the exact cut, closed, and strict
    multipliers, or ``None``.
    """
    singles = [entry for entry in pool if sum(entry[0]) + sum(entry[1]) == 1]
    if 0 < len(singles) < len(pool):
        fast = _cg_cut_witness_search(singles, closed_scaled, strict_scaled,
                                      variables, solver, exact, z3)
        if fast is not None:
            return fast
    return _cg_cut_witness_search(pool, closed_scaled, strict_scaled,
                                  variables, solver, exact, z3)


def _cg_cut_witness_search(pool, closed_scaled, strict_scaled, variables,
                           solver, exact, z3):
    """One witness LP over ``pool`` plus the premise rows, with a retry.

    The LP is built identically for the first attempt under the caller's
    solver budget and — when that attempt returns ``unknown`` rather than
    ``unsat`` — once more under the raised module budget. Both CG mints
    that reach this search are integer-domain gated, so the retry can
    construct the QF_LIA solver directly.
    """
    def build(target):
        cut_multipliers = [z3.Real(f"cg_cut_{index}") for index in range(len(pool))]
        closed_multipliers = [z3.Real(f"cg_closed_{index}") for index in range(len(closed_scaled))]
        strict_multipliers = [z3.Real(f"cg_strict_{index}") for index in range(len(strict_scaled))]
        target.add(*(value >= 0 for value
                     in cut_multipliers + closed_multipliers + strict_multipliers))
        target.add(z3.Sum(cut_multipliers) > exact(Fraction()))
        for position in range(len(variables)):
            target.add(
                sum((value * exact(Fraction(cut[0][position]))
                     for value, (_, _, cut) in zip(cut_multipliers, pool)),
                    exact(Fraction()))
                + sum((value * exact(Fraction(coefficients[position]))
                       for value, (coefficients, _) in zip(closed_multipliers, closed_scaled)),
                      exact(Fraction()))
                + sum((value * exact(Fraction(coefficients[position]))
                       for value, (coefficients, _) in zip(strict_multipliers, strict_scaled)),
                      exact(Fraction()))
                == exact(Fraction()))
        constant_expr = sum((value * exact(Fraction(cut[1]))
                             for value, (_, _, cut) in zip(cut_multipliers, pool)),
                            exact(Fraction())) \
            + sum((value * exact(Fraction(bound))
                   for value, (_, bound) in zip(closed_multipliers, closed_scaled)),
                  exact(Fraction())) \
            + sum((value * exact(Fraction(bound))
                   for value, (_, bound) in zip(strict_multipliers, strict_scaled)),
                  exact(Fraction()))
        target.add(constant_expr <= exact(Fraction()))
        target.add(z3.Or(constant_expr < exact(Fraction()),
                         z3.Sum(strict_multipliers) > exact(Fraction())))
        return cut_multipliers, closed_multipliers, strict_multipliers

    witness_solver = solver()
    cut_multipliers, closed_multipliers, strict_multipliers = build(witness_solver)
    result = witness_solver.check()
    if result == z3.unknown:
        retry_solver = z3.SolverFor("QF_LIA")
        retry_solver.set(timeout=_CG_WITNESS_RETRY_TIMEOUT_MS,
                         rlimit=_CG_WITNESS_RETRY_RLIMIT)
        build(retry_solver)
        if retry_solver.check() == z3.sat:
            result = z3.sat
            witness_solver = retry_solver
    if result != z3.sat:
        return None
    model = witness_solver.model()
    return ([_z3_fraction(z3, model.eval(value, model_completion=True))
             for value in cut_multipliers],
            [_z3_fraction(z3, model.eval(value, model_completion=True))
             for value in closed_multipliers],
            [_z3_fraction(z3, model.eval(value, model_completion=True))
             for value in strict_multipliers])


def _cg_cut_infeasibility_certificate(statement, normalized, split, solver, exact, z3):
    """Search a Chvátal–Gomory cut refutation of integer feasibility.

    Runs after the bounded-box and congruence searches failed on an
    integer-domain statement whose real relaxation stays feasible: the
    band ``1/2 <= x + y <= 9/10`` has no pinning bounds and no effective
    equality, yet the single cut ``x + y <= floor(9/10) = 0`` combined
    with the original lower row ``-x - y <= -1/2`` is real-infeasible.
    The bounded pool of :func:`_cg_cut_pool` supplies candidate cuts;
    Z3 then searches for a Farkas/Motzkin witness over the cuts and the
    premise rows — nonnegative multipliers, zero coefficients,
    nonpositive constant, and either a negative constant or a positive
    strict weight. At least one cut must carry weight: a witness over
    the premise rows alone is real infeasibility, which the Farkas
    layer already owns. Single-round CG is not complete; when this search
    finds no witness, :func:`_cg_iterated_cut_infeasibility_certificate`
    feeds the single-row cuts back and attempts a two-round closure.
    The verifier re-derives every cut from its declared multipliers, so
    nothing about the pool is trusted.
    """
    if normalized["domain"] != "integer":
        return None
    closed_rows, strict_rows = split
    variables = normalized["variables"]
    closed_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in closed_rows]
    strict_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in strict_rows]
    pool = _cg_cut_pool(closed_scaled, strict_scaled)
    if not pool:
        return None
    witness = _cg_cut_witness(pool, closed_scaled, strict_scaled,
                              variables, solver, exact, z3)
    if witness is None:
        return None
    exact_cut, exact_closed, exact_strict = witness
    used = [index for index, value in enumerate(exact_cut) if value != 0]
    if not used:
        return None
    cuts = []
    used_witness = []
    combined_constant = Fraction(0)
    for index in used:
        closed_mults, strict_mults, cut = pool[index]
        cuts.append({"closed_multipliers": [_render_fraction(Fraction(value))
                                            for value in closed_mults],
                     "strict_multipliers": [_render_fraction(Fraction(value))
                                            for value in strict_mults],
                     "derived_coefficients": {name: _render_fraction(Fraction(cut[0][position]))
                                              for position, name in enumerate(variables)},
                     "derived_constant": _render_fraction(Fraction(cut[1]))})
        used_witness.append(exact_cut[index])
        combined_constant += exact_cut[index] * cut[1]
    for value, (_, bound) in zip(exact_closed, closed_scaled):
        if value:
            combined_constant += value * bound
    for value, (_, bound) in zip(exact_strict, strict_scaled):
        if value:
            combined_constant += value * bound
    certificate = {"schema_version": 1, "kind": "integer_cg_cut_infeasibility",
                   "statement_hash": _statement_hash(statement),
                   "closed_premise_rows": [row[0] for row in closed_rows],
                   "strict_premise_rows": [row[0] for row in strict_rows],
                   "cuts": cuts,
                   "cut_multipliers": [_render_fraction(value) for value in used_witness],
                   "closed_multipliers": [_render_fraction(value) for value in exact_closed],
                   "strict_multipliers": [_render_fraction(value) for value in exact_strict],
                   "combined_constant": _render_fraction(combined_constant)}
    verify_integer_cg_cut_infeasibility(statement, certificate)
    return certificate


def _cg_iterated_cut_infeasibility_certificate(statement, normalized, split, solver, exact, z3):
    """Search a two-round Chvátal–Gomory cut refutation of integer feasibility.

    Runs when the single-round search of
    :func:`_cg_cut_infeasibility_certificate` found no witness: the
    single-row cuts of the premise rows are fed back as closed rows
    (focused feedback, at most one cut per premise row) and the staged
    pool of :func:`_cg_cut_pool` is rebuilt over the merged row space.
    Rank-two classes such as the diamond bands ``1/2 <= x + y <= 3/2`` and
    ``-1/2 <= x - y <= 1/2`` with one further free variable stay
    real-feasible after every rank-one cut, yet ``x <= 0`` and ``x >= 1``
    combine the fed-back band cuts into a refutation. The iteration is
    attempted while the merged row space stays inside
    :data:`_CG_ITERATED_ROW_LIMIT`; the round-two pool is the full staged
    enumeration while it fits :data:`_CG_COMBINATION_ROW_LIMIT` and the
    focused enumeration of :func:`_cg_focused_cut_pool` (combinations
    drawn only from the round-one cut rows) above that. The certificate
    records every
    derivation — unit multipliers for the round-one cuts, integer
    multipliers over the merged space for the round-two cuts, and the
    witness over round-two cuts, round-one rows, and premise rows; the
    verifier re-derives the whole chain, so nothing about the pool is
    trusted.
    """
    if normalized["domain"] != "integer":
        return None
    closed_rows, strict_rows = split
    variables = normalized["variables"]
    closed_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in closed_rows]
    strict_scaled = [_cg_scale_row(coefficients, constant, variables)
                     for _, coefficients, constant in strict_rows]
    premise_rows = ([(coefficients, bound, False) for coefficients, bound in closed_scaled]
                    + [(coefficients, bound, True) for coefficients, bound in strict_scaled])
    round1 = []
    for index in range(len(premise_rows)):
        multipliers = [0] * len(premise_rows)
        multipliers[index] = 1
        cut = _cg_rounded_cut(premise_rows, multipliers)
        if cut is not None:
            round1.append(([1 if position == index else 0
                            for position in range(len(closed_scaled))],
                           [1 if position == index - len(closed_scaled) else 0
                            for position in range(len(strict_scaled))],
                           cut))
    if not round1 or len(round1) + len(premise_rows) > _CG_ITERATED_ROW_LIMIT:
        return None
    merged_closed = [(cut[0], cut[1]) for _, _, cut in round1] + list(closed_scaled)
    if len(merged_closed) + len(strict_scaled) <= _CG_COMBINATION_ROW_LIMIT:
        pool = _cg_cut_pool(merged_closed, strict_scaled)
    else:
        pool = _cg_focused_cut_pool(merged_closed, strict_scaled, len(round1))
    if not pool:
        return None
    witness = _cg_cut_witness(pool, merged_closed, strict_scaled,
                              variables, solver, exact, z3)
    if witness is None:
        return None
    exact_cut, exact_merged, exact_strict = witness
    used = [index for index, value in enumerate(exact_cut) if value != 0]
    if not used:
        return None
    cuts = []
    used_witness = []
    combined_constant = Fraction(0)
    for index in used:
        closed_mults, strict_mults, cut = pool[index]
        cuts.append({"closed_multipliers": [_render_fraction(Fraction(value))
                                            for value in closed_mults],
                     "strict_multipliers": [_render_fraction(Fraction(value))
                                            for value in strict_mults],
                     "derived_coefficients": {name: _render_fraction(Fraction(cut[0][position]))
                                              for position, name in enumerate(variables)},
                     "derived_constant": _render_fraction(Fraction(cut[1]))})
        used_witness.append(exact_cut[index])
        combined_constant += exact_cut[index] * cut[1]
    for value, (_, bound) in zip(exact_merged, merged_closed):
        if value:
            combined_constant += value * bound
    for value, (_, bound) in zip(exact_strict, strict_scaled):
        if value:
            combined_constant += value * bound
    certificate = {"schema_version": 1,
                   "kind": "integer_iterated_cg_cut_infeasibility",
                   "statement_hash": _statement_hash(statement),
                   "closed_premise_rows": [row[0] for row in closed_rows],
                   "strict_premise_rows": [row[0] for row in strict_rows],
                   "round1_cuts": [{"closed_multipliers": [_render_fraction(Fraction(value))
                                                           for value in closed_mults],
                                    "strict_multipliers": [_render_fraction(Fraction(value))
                                                           for value in strict_mults],
                                    "derived_coefficients": {name: _render_fraction(Fraction(cut[0][position]))
                                                             for position, name in enumerate(variables)},
                                    "derived_constant": _render_fraction(Fraction(cut[1]))}
                                   for closed_mults, strict_mults, cut in round1],
                   "cuts": cuts,
                   "cut_multipliers": [_render_fraction(value) for value in used_witness],
                   "round1_multipliers": [_render_fraction(value)
                                          for value in exact_merged[:len(round1)]],
                   "closed_multipliers": [_render_fraction(value)
                                          for value in exact_merged[len(round1):]],
                   "strict_multipliers": [_render_fraction(value) for value in exact_strict],
                   "combined_constant": _render_fraction(combined_constant)}
    verify_integer_iterated_cg_cut_infeasibility(statement, certificate)
    return certificate


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
        elif kind == "integer_cg_cut_infeasibility":
            verify_integer_cg_cut_infeasibility(statement, certificate)
            return "unresolved", "exact_integer_cg_cut_infeasibility/v1", {
                "statement_hash": _statement_hash(statement), "backend": "portable_certificate",
                "logic": "QF_LIA",
                "premises_consistent": False, "vacuous_implication_rejected": True,
                "certificate_proof_domain": "integer",
                "portable_certificate": certificate, "portable_certificate_verified": True}
        elif kind == "integer_iterated_cg_cut_infeasibility":
            verify_integer_iterated_cg_cut_infeasibility(statement, certificate)
            return "unresolved", "exact_integer_iterated_cg_cut_infeasibility/v1", {
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

        def try_cg_cut_infeasibility():
            if normalized["domain"] != "integer" or split is None \
                    or "portable_certificate" in evidence:
                return
            certificate = _cg_cut_infeasibility_certificate(
                statement, normalized, split, solver, exact, z3)
            checker_name = "exact_integer_cg_cut_infeasibility/v1"
            if certificate is None:
                certificate = _cg_iterated_cut_infeasibility_certificate(
                    statement, normalized, split, solver, exact, z3)
                checker_name = "exact_integer_iterated_cg_cut_infeasibility/v1"
            if certificate is not None:
                evidence.update({"portable_certificate": certificate,
                                 "portable_certificate_checker": checker_name,
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
            try_cg_cut_infeasibility()
            return "unresolved", checker, evidence
        try_farkas_inconsistency()
        try_bounded_integer_infeasibility()
        try_integer_congruence_infeasibility()
        try_cg_cut_infeasibility()
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
