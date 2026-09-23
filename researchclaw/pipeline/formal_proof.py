"""Bounded professional-solver checks for typed linear arithmetic.

The input is data, not solver code: coefficients are exact rationals and the
grammar contains no expressions, quantifiers, functions, or Python evaluation.
"""
from __future__ import annotations

import re
from fractions import Fraction
from itertools import product

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
        if isinstance(certificate, dict) and certificate.get("kind") == "exhaustive_bounded_integer_implication":
            verify_exhaustive_integer_certificate(statement, certificate)
            checker_name = "exact_bounded_integer_enumeration/v1"
            proof_domain = "bounded_integer"
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
    evidence.update({"implication_holds": False, "counterexample": rendered,
                     "counterexample_exactly_validated": True})
    return "disproved", checker, evidence
