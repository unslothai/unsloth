# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keeping private data out of Deep Research web queries and prompts.

Search queries leave the machine, so credentials, personal identifiers, and private network
addresses are stripped before a query is issued. Prompt-delimiter tags in gathered evidence
are escaped so untrusted text cannot close a wrapper block and inject instructions, and a
source URL cannot close its own citation to inject a link.
"""

from __future__ import annotations

import difflib
import ipaddress
import re
from urllib.parse import quote, quote_plus


# Any occurrence inside untrusted evidence is escaped so gathered content cannot close a prompt block early.
_PROMPT_DELIMITER_TAGS = re.compile(
    r"</?\s*(?:untrusted_web_evidence|untrusted_evidence|source_catalog"
    r"|document_source_catalog|conversation_context_json|research_question"
    r"|approved_plan|untrusted_research_state_json|research_state_json"
    r"|untrusted_query_history_json|query_history_json"
    r"|untrusted_synthesis_audit_json|synthesis_audit_json)\s*>",
    re.IGNORECASE,
)
# Escaped so evidence cannot move the report boundary; spelled out to avoid importing prompts.
_REPORT_BOUNDARY_TAG = re.compile(r"<!--\s*UNSLOTH_FINAL_REPORT\s*-->")
# RFC 9110 11.4: credentials = auth-scheme [ 1*SP ( token68 / #auth-param ) ].
# IANA registry plus vendor schemes; not OAuth/Mutual, mostly prose ("OAuth 2.0", "Mutual TLS").
_REGISTERED_AUTH_SCHEMES = frozenset(
    {
        "aws4-hmac-sha256",
        "basic",
        "bearer",
        "concealed",
        "digest",
        "dpop",
        "gnap",
        "hawk",
        "hoba",
        "negotiate",
        "ntlm",
        "privatetoken",
        "scram-sha-1",
        "scram-sha-256",
        "sharedkey",
        "sharedkeylite",
        "token",
        "vapid",
    }
)
_SCHEME_ALTERNATION = "|".join(
    sorted(map(re.escape, _REGISTERED_AUTH_SCHEMES), key = len, reverse = True)
)
# Rich-text pastes carry no-break and thin spaces where a header has plain ones.
_HSPACE = "[ \\t\u00a0\u2000-\u200a\u202f\u205f\u3000]"
_AUTH_PARAM = r"""(?:[\w!#$%&'*+.^`|~-]+=(?:[ \t]+(?=\\?["'“‘]|[^\s,"]+[ \t]*,[ \t]*[\w!#$%&'*+.^`|~-]+[ \t]*=)|(?:(?<=key=)|(?<=token=)|(?<=secret=)|(?<=password=)|(?<=credential=)|(?<=signature=)|(?<=nonce=)|(?<=response=))(?:[ \t]*\r?\n)?[ \t]+)?|[\w!#$%&'*+.^`|~-]+[ \t]+=[ \t]*|(?=[A-Za-z_-]{1,15}\*?=)(?-i:[A-Z]?[a-z][a-z_-]*(?:[A-Z][a-z_-]+)*|[A-Z][A-Z_-]*)\*?=[ \t]*)(?:\\"[^"\r\n]*?\\"|"(?:[^"\\\r\n]|\\.)*"|'[^'\r\n]*'|“[^”\r\n]*”|‘[^’\r\n]*’|[^\s\u00a0\u2000-\u200a\u202f\u205f\u3000,"=()<>\[\]{}”’][^\s\u00a0\u2000-\u200a\u202f\u205f\u3000,"()<>\[\]{}”’]*)""".replace(
    r"[ \t]", _HSPACE
)
_QUERY_AUTHORIZATION = re.compile(
    rf"""(?aix)(?<![A-Za-z0-9])(?:(?<=proxy-)|(?<=x-)|(?<=http_)|(?<=x_)|(?<![a-z][-_]))authorization(?:[ \t_-]*header(?:[ \t_-]*value)?)?(?:\\?["'”’]\]?[ \t]*(?:=>|[:=,])(?:[ \t]*(?:\r?\n[ \t]*)?\\?["'“‘]value\\?["'”’][ \t]*[:=])?|\]?[*_~`]*[ \t]*(?:=>|[:=])[*_~`]*)(?:[ \t]*\r?\n)?[ \t]*(?:\[[ \t]*(?:\r?\n[ \t]*)?)?\\?["'“‘]?(?P<scheme>[\w!#$%&'*+.^`|~-]+)(?:[ \t]+|[ \t]*\r?\n[ \t]+)[*_~`"'“‘]*
    (?P<credentials>(?:(?<=basic\s)|(?<=bearer\s)|(?<=negotiate\s)|(?<=ntlm\s))(?:[\w.~+/-]|\\/)+=*(?![\w"'“‘])
    |(?:,[ \t]*)*{_AUTH_PARAM}(?:(?:[ \t]*,)+[ \t]*(?:\r?\n[ \t]*)?{_AUTH_PARAM})*|[\w.~+/-]+(?::[\w.~+/-]+)*=*)
    (?P<more>(?:[ \t]*,[ \t]*(?:\r?\n[ \t]*)?(?:
    (?:{_SCHEME_ALTERNATION}
    |[\w-]*(?:key|token|secret|password|credential|signature)|(?P=scheme))(?:[ \t]+|[ \t]*\r?\n[ \t]+)
    (?:[\w.~+/-]+(?::[\w.~+/-]+)*=*(?=[\s,"'`)\]}}>”’.;!?—–]|$)(?![ \t]*=)(?!(?<==)["'“‘][^"'”’\s,]*["'”’])|{_AUTH_PARAM}(?:(?:[ \t]*,)+[ \t]*{_AUTH_PARAM})*)
    |(?!(?:HTTP|GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|CONNECT|TRACE)[ \t])(?:(?-i:[A-Z][A-Za-z0-9]*[-\d][\w!#$%&'*+.^`|~-]*|[A-Z]{{2,}}(?=[ \t]+[\w!#$%&'*+.^`|~-]*(?:key|token|secret|password|credential|signature)\*?[ \t]*=)))[ \t]+{_AUTH_PARAM}(?:(?:[ \t]*,)+[ \t]*{_AUTH_PARAM})*
    |(?!(?:HTTP|GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|CONNECT|TRACE)[ \t])(?-i:[A-Z][A-Za-z0-9-]*[ \t]+(?![\w.~+/-]+=+[^\s,=])(?=[\w.~+-][\w.~+/-]{{7}})(?=[\w.~+/:-]*(?:[\d+/=]|[a-z][A-Z]|[A-Z]{{2}})|[\w.~+/-]{{16}})[\w.~+/-]+(?::[\w.~+/-]+)*=*)))*)""".replace(
        r"[ \t]", _HSPACE
    )
)
_REVERSED_HAR_HEADER = re.compile(
    r"""(?i)(?P<value>\\?"value\\?"\s*:\s*\\?"(?:[^"\\]|\\.)*?\\?")(?P<sep>\s*,\s*)"""
    r"""(?P<name>\\?"name\\?"\s*:\s*\\?"(?:proxy-|x-)?authorization\\?")"""
)
_QUERY_AUTHORIZATION_ARRAY = re.compile(
    r"""(?i)(?<![A-Za-z0-9])authorization(?:\\?["'”’]\]?\s*(?:=>|[:=,])|\]?\s*(?:=>|[:=]))\s*\[(?P<members>[^\[\]]*)\]"""
)
_ARRAY_MEMBER = re.compile(
    r"""(?<!\\)"((?:[^"\\]|\\.)*)"|\\"((?:[^"\\]|\\.)*?)\\"|'([^']*)'|“([^”]*)”|‘([^’]*)’"""
)
_AUTH_PARAM_PAIR = re.compile(
    r"""([\w!#$%&'*+.^`|~-]+)[ \t]*=(?:[ \t]*\r?\n)?[ \t]*(\\"[^"\r\n]*?\\"|"(?:[^"\\\r\n]|\\.)*"|'[^'\r\n]*'|“[^”\r\n]*”|‘[^’\r\n]*’|[^\s\u00a0\u2000-\u200a\u202f\u205f\u3000,]+)""".replace(
        r"[ \t]", _HSPACE
    )
)
# Digest parameters that describe the challenge rather than prove identity.
_PUBLIC_AUTH_PARAMS = frozenset(
    {"algorithm", "charset", "domain", "nc", "qop", "realm", "stale", "uri", "userhash"}
)
# Prose after "Authorization:" ("design patterns", "RBAC role=admin") is not a credential.
_CREDENTIAL_SHAPE = re.compile(r"(?=.{4})(?:.*(?:[\d+/=]|[a-z][A-Z]|[A-Z]{2})|.{16})")
_AUTH_SECRET_PARAM_SUFFIXES = (
    "credential",
    "key",
    "nonce",
    "password",
    "response",
    "secret",
    "signature",
    "token",
)
_REGISTERED_SCHEME_NAMES = frozenset(
    re.sub(r"[^a-z0-9]", "", name) for name in _REGISTERED_AUTH_SCHEMES
)
_COALESCED_MEMBER = re.compile(
    rf"\s*,\s*(?=(?:{_SCHEME_ALTERNATION}|[\w-]*(?:key|token|secret|password|credential|signature))\s"
    r"|(?!(?-i:HTTP|GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|CONNECT|TRACE)\s)(?-i:[A-Z][A-Za-z0-9]*[-\d][\w!#$%&'*+.^`|~-]*|[A-Z]{2,}(?=\s+[\w!#$%&'*+.^`|~-]*(?:key|token|secret|password|credential|signature)\*?\s*=))\s+[\w!#$%&'*+.^`|~-]+\s*="
    r"|(?!(?-i:HTTP|GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|CONNECT|TRACE)\s)(?-i:[A-Z][A-Za-z0-9-]*\s+(?![\w.~+/-]+=+[^\s,=])(?=[\w.~+-][\w.~+/-]{7})(?=[\w.~+/:-]*(?:[\d+/=]|[a-z][A-Z]|[A-Z]{2})|[\w.~+/-]{16})))",
    re.IGNORECASE,
)
_TOKEN_PUNCTUATION = "\"'`,;:()[]{}<>“”‘’"
_QUERY_CREDENTIAL = re.compile(
    r"""(?ix)(?<![A-Za-z0-9])(?:api[\s_-]?key|access[\s_-]?(?:key|token)
    |auth[\s_-]?token|bearer[\s_-]?token|client[\s_-]?secret|private[\s_-]?key
    |refresh[\s_-]?token|session[\s_-]?token|authorization|password|secret|token)\s*[:=]\s*
    (?:"[^"]*"|'[^']*'|“[^”]*”|‘[^’]*’|[^\s,;]+)"""
)
_QUERY_NAMED_ASSIGNMENT = re.compile(
    r"""(?x)(?<![A-Za-z0-9])(?P<label>[A-Za-z][A-Za-z0-9_-]{0,100})\s*[:=]\s*
    (?P<value>"[^"]*"|'[^']*'|“[^”]*”|‘[^’]*’|[^\s,;]+)"""
)
_QUERY_CREDENTIAL_SUFFIXES = (
    "apikey",
    "accesskey",
    "accesstoken",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "privatekey",
    "refreshtoken",
    "secretkey",
    "sessiontoken",
    "authorization",
    "password",
    "token",
)
_QUERY_PUBLIC_ASSIGNMENT_SUFFIXES = ("designtoken", "cancellationtoken")
# Bearer tokens carry no key=value label; the length floor keeps ordinary prose ("bearer of bad news") from matching.
_QUERY_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_QUERY_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_QUERY_PRIVATE_ID = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_QUERY_OPAQUE_TOKEN = re.compile(
    r"\b(?:eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r"|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{16,}"
    r"|hf_(?:oauth_[A-Za-z0-9._~+/=-]{20,}|[A-Za-z0-9]{20,})|glpat-[A-Za-z0-9_-]{20,}"
    r"|AKIA[A-Z0-9]{16})\b"
)
# Requires separators or '+' so bare numeric research terms are not redacted.
_QUERY_PHONE = re.compile(
    r"(?<!\w)\+\d[\d\s().-]{7,17}\d(?!\w)|(?<!\w)\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)"
)
_QUERY_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_QUERY_IPV6 = re.compile(
    r"(?<![0-9A-Fa-f:])\[?(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f.]*(?:%[A-Za-z0-9_.-]+)?\]?"
    r"(?![0-9A-Fa-f:])"
)
_QUERY_LABELED_PRIVATE_ID = re.compile(
    r"(?ix)\b(?:passport|driver(?:'s)?[\s_-]?licen[cs]e|national[\s_-]?id"
    r"|tax[\s_-]?id|account[\s_-]?(?:number|no))\s*[:=#-]?\s*[A-Za-z0-9][A-Za-z0-9_-]{4,24}\b"
)
_QUERY_PAYMENT_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _luhn_valid(candidate: str) -> bool:
    digits = [int(character) for character in candidate if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _redact_nonpublic_ip(match: "re.Match[str]") -> str:
    try:
        return " " if not ipaddress.ip_address(match.group(0)).is_global else match.group(0)
    except ValueError:
        return match.group(0)


def _redact_nonpublic_ipv6(match: "re.Match[str]") -> str:
    # Strip brackets and any zone id before validating; redact non-global addresses.
    candidate = match.group(0).strip("[]").split("%", 1)[0]
    try:
        return " " if not ipaddress.ip_address(candidate).is_global else match.group(0)
    except ValueError:
        return match.group(0)


def _escape_link_destination(url: str) -> str:
    # Escape an unbalanced ")" so a source URL cannot close the citation and inject a link.
    out: list[str] = []
    depth = 0
    for char in url:
        if char == "\\":
            out.append("\\\\")
        elif char == "(":
            depth += 1
            out.append(char)
        elif char == ")" and depth == 0:
            out.append("\\)")
        else:
            if char == ")":
                depth -= 1
            out.append(char)
    return "".join(out)


def _shield_untrusted(text: str) -> str:
    """Escape prompt-delimiter tags and the report boundary marker in untrusted evidence, so
    gathered content cannot close a wrapper block to inject model instructions, nor move the
    boundary that selects the published report."""
    if not text:
        return text

    def escape(match: re.Match) -> str:
        return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

    return _REPORT_BOUNDARY_TAG.sub(escape, _PROMPT_DELIMITER_TAGS.sub(escape, text))


def _escape_echo(text: str) -> str:
    return re.sub(
        r"%([0-9A-Fa-f])([0-9A-Fa-f])",
        lambda m: "%"
        + "".join(f"[{c.upper()}{c.lower()}]" if c.isalpha() else c for c in m.groups()),
        re.escape(text),
    )


def _authorization_values(match: re.Match) -> list[str]:
    """Decoded credential values of an Authorization match; empty when it reads as prose."""
    found = match["credentials"]
    if re.fullmatch(r"(?:[\w.~+/-]|\\/)+(?::[\w.~+/-]+)*=*", found):
        pairs = [("", found)]
    else:
        pairs = [
            (name.lower(), value)
            for name, value in _AUTH_PARAM_PAIR.findall(found)
            if name.lower() not in _PUBLIC_AUTH_PARAMS
        ]
    raw = [value.strip("\\\"'“”‘’") for _, value in pairs]
    pairs = [(name, re.sub(r"\\(.)", r"\1", value.strip("\\\"'“”‘’"))) for name, value in pairs]
    values = [value for _, value in pairs] + [value for value in raw if "\\" in value]
    coalesced = []
    repeated = rf"\s*,\s*(?={re.escape(match['scheme'])}\s)"
    members = _COALESCED_MEMBER.split(match["more"] or "")
    members = [piece for member in members for piece in re.split(repeated, member, flags = re.I)]
    for member in members[1:]:
        parsed = _QUERY_AUTHORIZATION.match("Authorization: " + member)
        coalesced += _authorization_values(parsed) if parsed else []
    scheme = re.sub(r"[^a-z0-9]", "", match["scheme"].lower())
    if (
        scheme in _REGISTERED_SCHEME_NAMES
        or (
            scheme.endswith(_AUTH_SECRET_PARAM_SUFFIXES)
            and not scheme.endswith(_QUERY_PUBLIC_ASSIGNMENT_SUFFIXES)
        )
    ) or any(
        (
            name.endswith(_AUTH_SECRET_PARAM_SUFFIXES)
            and not re.sub(r"[^a-z0-9]", "", name).endswith(_QUERY_PUBLIC_ASSIGNMENT_SUFFIXES)
        )
        or _CREDENTIAL_SHAPE.match(value)
        for name, value in pairs
    ):
        # "dXNl...= server" also parses as a parameter; keep the padded token68 itself.
        padded = [piece for piece in found.split() if re.fullmatch(r"[\w.~+/-]+=+", piece)]
        return values + padded + coalesced
    return coalesced


def _array_values(match: re.Match) -> list[str]:
    values = []
    members = ["".join(groups) for groups in _ARRAY_MEMBER.findall(match["members"])]
    # Unquoted lists, as Java/Spring prints them: {Authorization=[Basic a, Basic b]}.
    for text in members or [member.strip() for member in match["members"].split(",")]:
        member = _QUERY_AUTHORIZATION.match("Authorization: " + text)
        if member:
            values += _authorization_values(member)
    return values


def _sanitize_public_query(query: str) -> str:
    # Keep only tokens the plain passes also keep, so this can never remove less than they do.
    query = _REVERSED_HAR_HEADER.sub(r"\g<name>\g<sep>\g<value>", query)
    plain = _redact_public_query(query, authorization = False)
    with_authorization = _redact_public_query(query, authorization = True)
    credentials, exact = set(), set()
    found = [
        value
        for match in _QUERY_AUTHORIZATION.finditer(query)
        for value in _authorization_values(match)
    ]
    found += [
        value
        for match in _QUERY_AUTHORIZATION_ARRAY.finditer(query)
        for value in _array_values(match)
    ]
    for value in found:
        trimmed = value.rstrip(_TOKEN_PUNCTUATION + ".!?")
        for variant in {value, trimmed, trimmed.rstrip("=")}:
            exact.update(
                {
                    quote(variant, safe = "", errors = "surrogatepass"),
                    quote(variant, errors = "surrogatepass"),
                    quote_plus(variant, errors = "surrogatepass"),
                }
            )
            exact.update({variant, variant.replace("/", "\\/")})
            credentials.update(piece for piece in [variant, *variant.split()] if len(piece) >= 8)
    exact.discard("")
    plain = [token for token in plain if not any(piece in token for piece in credentials)]
    tokens = plain
    if with_authorization != plain:
        # 600 tokens cover the 500-character output; the matcher is quadratic past that.
        plain, with_authorization = plain[:600], with_authorization[:600]
        matcher = difflib.SequenceMatcher(None, plain, with_authorization, autojunk = False)
        tokens = [
            token for a, _, size in matcher.get_matching_blocks() for token in plain[a : a + size]
        ]
    query = " ".join(tokens)
    if exact:
        values = sorted(exact, key = len, reverse = True)
        echo = "|".join(r"\s+".join(map(_escape_echo, value.split())) for value in values)
        query = re.sub(
            rf"(?:(?<![\w.~+-])|(?<=%[0-9A-Fa-f]{{2}}))(?:{echo})(?![\w~+-])", " ", query
        )
    query = " ".join(query.split()).strip(" ,;:-")[:500]
    if not any(character.isalnum() for character in query):
        raise ValueError("Research query contained only private or credential-like data")
    return query


def _redact_public_query(query: str, *, authorization: bool) -> list[str]:
    def redact_named_assignment(match: re.Match) -> str:
        label = re.sub(r"[^a-z0-9]", "", match.group("label").lower())
        if label.endswith(_QUERY_CREDENTIAL_SUFFIXES) and not label.endswith(
            _QUERY_PUBLIC_ASSIGNMENT_SUFFIXES
        ):
            return " "
        return match.group(0)

    if authorization:
        query = _QUERY_AUTHORIZATION_ARRAY.sub(
            lambda match: " " if _array_values(match) else match[0], query
        )
        query = _QUERY_AUTHORIZATION.sub(
            lambda match: " " if _authorization_values(match) else match[0], query
        )
    query = _QUERY_CREDENTIAL.sub(" ", query)
    query = _QUERY_NAMED_ASSIGNMENT.sub(redact_named_assignment, query)
    query = _QUERY_BEARER.sub(" ", query)
    query = _QUERY_EMAIL.sub(" ", query)
    query = _QUERY_PRIVATE_ID.sub(" ", query)
    query = _QUERY_OPAQUE_TOKEN.sub(" ", query)
    query = _QUERY_PHONE.sub(" ", query)
    query = _QUERY_LABELED_PRIVATE_ID.sub(" ", query)
    query = _QUERY_IPV4.sub(_redact_nonpublic_ip, query)
    query = _QUERY_IPV6.sub(_redact_nonpublic_ipv6, query)
    query = _QUERY_PAYMENT_CARD.sub(
        lambda match: " " if _luhn_valid(match.group(0)) else match.group(0),
        query,
    )
    return query.split()
