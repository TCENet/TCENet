import re
import sys
project_root_path = "/the_project_root_path/"
sys.path.append(project_root_path)

from util import re_auto, ioc_regex
primitives, regexes = ioc_regex.create_master_regexes_dict()
html_entity_pat = re.compile(r"(%s)" % re_auto.entities, re.I | re.X)
url_proto_pat = re.compile(r"""
    ^ %(protocol)s %(proto_sep)s
    """ % primitives, re.I | re.X)

def decode_entities(text):
    def subst(match):
        return re_auto.entity_map.get(match.group(1))

    return html_entity_pat.sub(subst, text)


def clean_observable(observable, rtype):
    ''' Transform an observable into a more clean, standardized form

    Args:
        observable (str): the string to clean.
        rtype (str): the type of observable (email, IP, etc.)
    '''
    try:
        observable, normalized = observable
        if normalized:
            return normalized
    except ValueError:
        pass
    observable = observable.strip()
    observable = re.sub(r"%(dot)s" % primitives, ".", observable, flags=re.X)
    observable = re.sub(r"%(at)s" % primitives, "@", observable, flags=re.X)
    observable = re.sub(r"%(colon)s" % primitives, ":", observable, flags=re.X)
    # lowercase non url & filepath matches
    if not rtype == "url" and not rtype == "filepath" \
            and not rtype == "filename" and not rtype == "cc" \
            and not rtype == "asn" and not rtype == "cve":
        observable = observable.lower()

        # trim user agent-strings
    if rtype == "useragent":
        # check for references to 'host:' and/or 'accept:'
        _hstndx = -1
        try:
            _hstndx = observable.index("host:")
            # check for 'accept'
            _acptndx = observable.index("accept:")
            if _acptndx >= 0 and _hstndx >= 0 and _acptndx < _hstndx:
                _hstndx = _acptndx
        except:
            pass
        if _hstndx >= 0:
            observable = observable[:_hstndx].strip()
    elif rtype == "url":
        observable = re.sub(r"^%(http)s%(proto_sep)s" % primitives,
                            "http://", observable, flags=re.I | re.X)
        observable = re.sub(r"^%(https)s%(proto_sep)s" % primitives,
                            "https://", observable, flags=re.I | re.X)
        observable = re.sub(r"^%(ftp)s%(proto_sep)s" % primitives,
                            "ftp://", observable, flags=re.I | re.X)
        observable = re.sub(r"^%(ftps)s%(proto_sep)s" % primitives,
                            "ftps://", observable, flags=re.I | re.X)
        observable = re.sub(r"^%(sftp)s%(proto_sep)s" % primitives,
                            "sftp://", observable, flags=re.I | re.X)
        if not url_proto_pat.search(observable):
            observable = "http://" + observable
    elif rtype == "cve":
        m = re.search(r"(\d+)-(\d+)", observable)
        observable = "CVE-%s-%s" % (m.group(1), m.group(2))
    elif rtype == "asn":
        m = re.search(r"(\d+(?:\.\d+)?)", observable)
        observable = m.group(1)
        if '.' in observable:
            hi, lo = observable.split('.')
            observable = str((65536 * int(hi)) + int(lo))
        observable = "AS" + observable
    elif rtype == "ipv4range" or rtype == "ipv6range":
        observable = re.sub("\s*-\s*", "-", observable)
    elif rtype == "isp" or rtype == "asnown":
        observable = observable.upper()
    elif rtype == "incident":
        observable = re.sub("\s*#\s*", " #", observable)
        if re.search("^INC\d+$", observable, re.I):
            m = re.search("(\d+)", observable)
            digits = m.group(0)
            observable = "INC" + "0" * (12 - len(digits)) + digits
        elif re.match(r"\d{4}-\d{4}-\d{3}", observable):
            observable = "CSIRT/NRC %s" % observable
        if re.match(r"(ticket|incident|bug|case)", observable, re.I):
            observable = observable.capitalize()
        else:
            observable = observable.upper()
    elif rtype == "cc":
        observable = re.sub(r"\s+", " ", observable)
        observable = re_auto.cc_map[observable.lower()]
    elif rtype == "topic":
        for topic, pat in re_auto.per_topic:
            if pat.search(observable):
                observable = topic
                break
    return observable

def extract_observables(text):
    """
    For each observable type, use the corresponding regex or
    callable to extract observables (such as IP addresses)
    from the given text and return their normalized forms.
    """
    results = {}
    results_origin = {}
    if not text:
        return results, results_origin
    try:
        text = text.read()
    except AttributeError:
        pass
    text = decode_entities(text)
    for typ, regex in regexes.items():
        if callable(regex):
            matches = regex(text)
        else:
            matches = regex.findall(text)
        observables = set()
        observables_origin = set()
        for match in matches:
            normalized = None
            try:
                observable, normalized = match

            except ValueError:
                observable = match
            if not observable:
                continue
            if not normalized:
                normalized = clean_observable(observable, typ)
            observables.add(normalized)
            observables_origin.add(observable)
        results[typ] = tuple(sorted(observables))
        results_origin[typ] = tuple(sorted(observables_origin))

    return results, results_origin