import gzip

from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List, Tuple

import xml.etree.ElementTree as ET

@dataclass
class InterproEntry:
    id: str
    element_type: str
    match_id: str
    element_name: str
    positions: List[Tuple[int]]

@dataclass
class Protein:
    uniprot_id: str
    name: str
    length: int
    parsed_entries: int
    total_entries: int
    entries: List[InterproEntry]

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

def parse_lines(lines: List[str]) -> Protein:
    return parse_one_protein(ET.fromstringlist(lines))

def parse_by_line(path, batch_size=1000):
    parsed = []
    with gzip.open(path, "rt") as fh:
        in_block = False
        prot_blocks = []
        for line in fh:
            if line.startswith("<protein "):
                in_block = True
                lines = []
            if in_block:
                lines.append(line)
            if line.startswith("</protein>"):
                in_block = False
                prot_blocks.append(lines)
                if len(prot_blocks) == batch_size:
                    parsed.extend(map(parse_lines, prot_blocks))
                    prot_blocks = []

        if len(prot_blocks) != 0:
            parsed.extend(map(parse_lines, prot_blocks))

    return parsed

def parse_by_line_multiproc(path, nprocs=2, batch_size=1000):
    parsed = []
    with Pool(processes=nprocs) as p:
        with gzip.open(path, "rt") as fh:
            in_block = False
            prot_blocks = []
            for line in fh:
                if line.startswith("<protein "):
                    in_block = True
                    lines = []
                if in_block:
                    lines.append(line)
                if line.startswith("</protein>"):
                    in_block = False
                    prot_blocks.append(lines)
                    if len(prot_blocks) == batch_size:
                        parsed.extend(p.map(parse_lines, prot_blocks))
                        prot_blocks = []

        if len(prot_blocks) != 0:
            parsed.extend(p.map(parse_lines, prot_blocks))

    return parsed

path = "/weka/scratch/weka/kellislab/rcalef/data/interpro/test_match_complete.large.xml.gz"
_ = parse_by_line(path)

