"""Run real OpenAI Agent against Day 6 pgvector/BM25/Hybrid retrieval."""
import argparse, json, os, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from agent_openai import OpenAIController, run_agent
from run_retrieval import load_db_chunks, DenseSearch, BM25Search, hybrid_rrf

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--questions",type=Path,default=Path("starter/data/converted/questions_agent.json"))
    p.add_argument("--split",choices=["dev","test"],default="dev")
    p.add_argument("--strategy",choices=["A","B","C"],default="B")
    p.add_argument("--retriever",choices=["dense","bm25","hybrid"],default="dense")
    p.add_argument("--top-k",type=int,default=3)
    p.add_argument("--candidate-k",type=int,default=20)
    p.add_argument("--max-searches",type=int,default=4)
    p.add_argument("--context-budget",type=int,default=1200)
    p.add_argument("--output",type=Path,default=Path("starter/results/agent"))
    a=p.parse_args()
    load_dotenv(Path("starter/.env")); load_dotenv()
    key=os.getenv("OPENAI_API_KEY",""); db=os.getenv("DATABASE_URL","")
    model=os.getenv("OPENAI_MODEL","gpt-6-luna")
    emb_model=os.getenv("EMBEDDING_MODEL","text-embedding-3-small")
    dims=int(os.getenv("EMBEDDING_DIMENSIONS","1536"))
    if not key or not db: raise RuntimeError("Thiếu OPENAI_API_KEY hoặc DATABASE_URL trong .env")
    qs=[q for q in json.loads(a.questions.read_text(encoding="utf-8")) if q["split"]==a.split]
    if not qs: raise ValueError("Không có câu hỏi cho split")
    chunks=load_db_chunks(db,a.strategy,dims)
    bm=BM25Search(chunks)
    dense=DenseSearch(db,a.strategy,dims)
    client=OpenAI(api_key=key,timeout=60,max_retries=2)
    def embed(q):
        return client.embeddings.create(model=emb_model,input=[q],dimensions=dims).data[0].embedding
    class Retriever:
        def search(self,query,top_k=3):
            if a.retriever=="bm25": return bm.search(query,top_k)
            d=dense.search(embed(query),a.candidate_k if a.retriever=="hybrid" else top_k)
            if a.retriever=="dense": return d[:top_k]
            b=bm.search(query,a.candidate_k)
            return hybrid_rrf(d,b,top_k)
    traces=[]
    try:
        for q in qs:
            controller=OpenAIController(key,model)
            try:
                result=run_agent(q["question"],controller,Retriever(),
                    max_searches=a.max_searches,top_k=a.top_k,budget=a.context_budget)
                traces.append({"query_id":q["id"],"split":a.split,"question":q["question"],
                    "answerable":q["answerable"],**result})
            except Exception as e:
                traces.append({"query_id":q["id"],"split":a.split,"question":q["question"],
                    "answerable":q["answerable"],"status":"error","error":repr(e),"events":[],"citations":[]})
            print(json.dumps(traces[-1],ensure_ascii=False))
    finally: dense.close()
    out=a.output/a.split/a.retriever; out.mkdir(parents=True,exist_ok=True)
    (out/"traces.json").write_text(json.dumps(traces,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary={"split":a.split,"retriever":a.retriever,"model":model,"strategy":a.strategy,
        "n":len(traces),"answered":sum(t["status"]=="answered" for t in traces),
        "abstained":sum(t["status"]=="abstain" for t in traces),
        "errors":sum(t["status"]=="error" for t in traces),
        "avg_elapsed_ms":round(sum(t.get("elapsed_ms",0) for t in traces)/len(traces),2)}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("\nSUMMARY",json.dumps(summary,ensure_ascii=False))
if __name__=="__main__": main()
