import gzip
import xml.etree.ElementTree as ET

from collections import defaultdict
from typing import Generator

from magneton.core_types import InterproEntry, Protein

# Details on InterPro XML format from Matthias Blum (reached via `interhelp@ebi.ac.uk`):
#
# We don't have a comprehensive documentation of match_complete.xml.gz,
# but basically it contains all member database matches found in InterPro.
#
# Each `<match>` element represents a hit or match reported by one of the
# InterPro member databases (Pfam, CDD, PANTHER, etc.). The attributes
# are:
# - id: member database accession, e.g. PF00155
# - name: member database name, e.g. Aminotransferase class I and II
# - dbname: member database, e.g. Pfam
# - status: legacy attribute, always equal to "T" (for "True"), we may
# remove it in the future
#     - model: most of the time equal to the value of the "id" attribute,
# but may contain extra information such as the subfamily accession in the
# case of PANTHER matches (the "id" attribute contains the family
# accession)
#     - evd: evidence, i.e. how the match was found

# A `<match>` element may contain an `<ipr>` element when the signature is
# integrated in an InterPro entry. The attributes of `<ipr>` elements are:
# - id: InterPro entry accession
# - name: InterPro entry name
# - type: InterPro entry type

# A `<match>` element always contain at least one `<lcn>` element, which
# represents a sequence hit. The attributes of `<lcn>` are:
# * start: start position of the hit on the sequence
# * end: end position of the hit on the sequence
# * fragments: individual boundaries of domains within the hit ,
# separated by commas. For instance "36-47-C,289-378-N" represents a
# discontinuous domain with two regions: 36-47 and 289-378
# * score: score, such as e-value, bitscore, or p-value reported by the
# member database
# * representative: boolean value that is equal to true if the match has
# been selected to be representative. Representative domains are selected
# to maximize coverage of a sequence while limiting overlap.

# Not all matches are eligible for being selected as representative. Before
# InterPro 102.0, only matches of type domain and repeat were eligible,
# and with InterPro 102.0 matches of type family are also eligible. For
# InterPro 103.0 we will add a "type" attribute to the `<match>` element.

def parse_one_protein(prot_ele: ET.Element) -> Protein:
    all_entries = [parse_one_match(x) for x in prot_ele.iter(tag="match")]
    parsed_entries = [x for x in all_entries if x]

    uniprot_id = prot_ele.get("id")
    name = prot_ele.get("name")
    uniprotkb_id = f"sp|{uniprot_id}|{name}"

    return Protein(
        uniprot_id=uniprot_id,
        kb_id=uniprotkb_id,
        name=prot_ele.get("name"),
        length=int(prot_ele.get("length")),
        entries=parsed_entries,
        parsed_entries=len(parsed_entries),
        total_entries=len(all_entries),
    )

def parse_one_match(match_ele: ET.Element) -> InterproEntry:
    grouped = defaultdict(list)
    for ele in match_ele.iter():
        grouped[ele.tag].append(ele)

    assert len(grouped["match"]) == 1
    match_obj = grouped["match"][0]

    if "ipr" in grouped:
        assert len(grouped["ipr"]) == 1
        ipr = grouped["ipr"][0]
    else:
        # If no "ipr" element, then this match doesn't have an InterPro ID
        # i.e. isn't integrated into InterPro.
        return {}

    # Want either all True or all False
    is_representative = [x.get("representative") for x in grouped["lcn"]]
    assert all(is_representative) or all([not x for x in is_representative])
    representative = is_representative[0] == 'true'

    positions = [(int(x.get("start")), int(x.get("end"))) for x in grouped["lcn"]]

    return InterproEntry(
        id=ipr.get("id"),
        element_type=ipr.get("type"),
        match_id=match_obj.get("id"),
        element_name=ipr.get("name"),
        representative=representative,
        positions=positions,
    )

def parse_from_xml(
    input_path: str,
    print_iter: int = 10000,
) -> Generator[Protein, None, None]:
    with gzip.open(input_path, "rt") as fh:
        in_block = False
        tot = 0
        for line in fh:
            if line.startswith("<protein "):
                in_block = True
                lines = []
            if in_block:
                lines.append(line)
            if line.startswith("</protein>"):
                in_block = False
                yield parse_one_protein(ET.fromstringlist(lines))
                tot += 1

                if tot % print_iter == 0:
                    print(f"Parsed proteins: {tot}", flush=True)
