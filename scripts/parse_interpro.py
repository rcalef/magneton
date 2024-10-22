import gzip
import pickle
import xml.etree.ElementTree as ET

import fire

from collections import defaultdict
from multiprocessing import Manager, Pool, Queue
from typing import List, BinaryIO

from magneton.types import InterproEntry, Protein

def parse_one_protein(prot_ele: ET.Element) -> Protein:
    all_entries = [parse_one_match(x) for x in prot_ele.iter(tag="match")]
    parsed_entries = [x for x in all_entries if x]
    return Protein(
        uniprot_id=prot_ele.get("id"),
        name=prot_ele.get("name"),
        length=prot_ele.get("length"),
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
        return {}

    return InterproEntry(
        id=ipr.get("id"),
        element_type=ipr.get("type"),
        match_id=match_obj.get("id"),
        element_name=ipr.get("name"),
        positions=[(int(x.get("start")), int(x.get("end"))) for x in grouped["lcn"]],
    )

def parse_and_write(
    input_path: str,
    output_path: str,
    print_iter: int = 10000,
):
    with (
        gzip.open(input_path, "rt") as fh,
        open(output_path, "wb") as out_fh,
        Pool(processes=2) as p,
    ):
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
                prot = parse_one_protein(ET.fromstringlist(lines))
                pickle.dump(prot, out_fh)
                tot += 1

                if tot % print_iter == 0:
                    print(f"Parsed proteins: {tot}")
    return

if __name__ == "__main__":
    fire.Fire(parse_and_write)
