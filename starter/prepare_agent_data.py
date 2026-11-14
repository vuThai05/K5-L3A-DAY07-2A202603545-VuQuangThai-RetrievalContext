"""Prepare Agent benchmark from original benchmark JSON files.
Accepts either a JSON array or an envelope with a `questions` array.
"""
import argparse, json
from pathlib import Path

def read_questions(path):
    obj=json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj,dict):
        obj=obj.get("questions")
    if not isinstance(obj,list):
        raise ValueError(f"{path}: cần JSON array hoặc object có trường questions dạng array")
    return obj

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--test",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cases-output",type=Path,default=None)
    a=p.parse_args()
    merged=[]
    for split,path in (("dev",a.dev),("test",a.test)):
        for q in read_questions(path):
            item=dict(q); item["split"]=split
            if item.get("answerable") is False:
                item["expected_behavior"]="search_and_abstain"
            merged.append(item)
    ids=[q.get("id") for q in merged]
    if any(not x for x in ids) or len(ids)!=len(set(ids)):
        raise ValueError("Question ID thiếu hoặc trùng")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("questions_agent:",len(merged),{s:sum(q["split"]==s for q in merged) for s in ("dev","test")})
    if a.cases_output:
        cases=[{"id":q["id"],"split":q["split"],"answerable":q["answerable"]} for q in merged]
        a.cases_output.parent.mkdir(parents=True,exist_ok=True)
        a.cases_output.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("cases:",len(cases))
if __name__=="__main__": main()
