import json
import asyncio
import os
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# 🔑 المفاتيح — حط مفاتيحك هنا بين علامتين التنصيص
# ============================================================
GEMINI_KEY = ""   # aistudio.google.com/app/apikey
GROQ_KEY   = ""   # console.groq.com/keys
TAVILY_KEY = ""   # tavily.com

PORT = 8000


# ============================================================
# 🧠 الموديلات (Fallback chain)
# ============================================================

MODELS = [
    {"id": "groq-llama", "name": "Llama 3.3 70B (Groq)",    "provider": "groq",   "model": "llama-3.3-70b-versatile", "priority": 1},
    {"id": "groq-8b",    "name": "Llama 3.1 8B (Groq)",     "provider": "groq",   "model": "llama-3.1-8b-instant",    "priority": 2},
    {"id": "gemini-25",  "name": "Gemini 2.5 Flash",        "provider": "gemini", "model": "gemini-2.5-flash",        "priority": 1},
    {"id": "gemini-38",  "name": "Gemini 3.8 Flash",        "provider": "gemini", "model": "gemini-3.8-flash",        "priority": 2},
]


def key_for(provider):
    return {"gemini": GEMINI_KEY, "groq": GROQ_KEY}.get(provider, "")


def refresh_availability():
    for m in MODELS:
        m["available"] = bool(key_for(m["provider"]).strip())


refresh_availability()


# ============================================================
# 💓 Keep-alive (لمنع نوم السيرفر على Render)
# ============================================================

SELF_URL = os.environ.get("RENDER_EXTERNAL_URL", "")


async def keep_alive():
    if not SELF_URL:
        return
    print(f"✅ Keep-alive: {SELF_URL}")
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await asyncio.sleep(300)
                r = await client.get(f"{SELF_URL}/healthz", timeout=30)
                print(f"💓 Ping: {r.status_code}")
            except Exception as e:
                print(f"❌ Ping: {e}")


# ============================================================
# 🔎 بحث ويب — Tavily
# ============================================================

async def tavily_search(query: str, max_results: int = 5):
    if not TAVILY_KEY.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_KEY.strip(),
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                },
            )
            if r.status_code != 200:
                return None
            data = r.json()
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": (item.get("content", "") or "")[:800],
                }
                for item in data.get("results", [])
            ]
    except Exception:
        return None


def format_search_context(results):
    if not results:
        return ""
    lines = ["[معلومات من بحث ويب حديث — استخدمها في الإجابة واذكر المصادر]"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n{i}. {r['title']}")
        lines.append(f"   الرابط: {r['url']}")
        lines.append(f"   المقتطف: {r['content']}")
    lines.append("\n[تعليمات: أجب بالاعتماد على المعلومات دي، واذكر روابط المصادر في آخر الإجابة]")
    return "\n".join(lines)


# ============================================================
# 🌐 المزودون
# ============================================================

class ProviderBusy(Exception):
    pass


async def stream_gemini(messages, model):
    key = GEMINI_KEY.strip()
    if not key:
        raise ProviderBusy("Gemini ناقص مفتاح")
    contents = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
    ]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model + ":streamGenerateContent?key=" + key + "&alt=sse"
    )
    payload = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.8},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload) as r:
            if r.status_code in (429, 500, 502, 503, 504):
                raise ProviderBusy(f"Gemini {r.status_code}")
            if r.status_code != 200:
                body = await r.aread()
                raise Exception(body.decode(errors="ignore"))
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for c in j.get("candidates", []):
                    for p in c.get("content", {}).get("parts", []):
                        if p.get("text"):
                            yield p["text"]


async def stream_groq(messages, model):
    key = GROQ_KEY.strip()
    if not key:
        raise ProviderBusy("Groq ناقص مفتاح")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0.8,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code in (429, 500, 502, 503, 504):
                raise ProviderBusy(f"Groq {r.status_code}")
            if r.status_code != 200:
                body = await r.aread()
                raise Exception(body.decode(errors="ignore"))
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = j.get("choices", [{}])[0].get("delta", {})
                txt = delta.get("content")
                if txt:
                    yield txt


async def stream_chat(messages, model_id):
    for m in MODELS:
        if m["id"] != model_id:
            continue
        if m["provider"] == "gemini":
            async for tok in stream_gemini(messages, m["model"]):
                yield tok
            return
        if m["provider"] == "groq":
            async for tok in stream_groq(messages, m["model"]):
                yield tok
            return
    raise Exception("الموديل غير موجود: " + model_id)


def fallback_chain(selected_id):
    available = [m for m in MODELS if m.get("available")]
    available.sort(key=lambda x: (x["id"] != selected_id, x["priority"]))
    return available


# ============================================================
# 🚀 API
# ============================================================

class Msg(BaseModel):
    role: str
    content: str


class Req(BaseModel):
    messages: list[Msg]
    model: str = "groq-llama"
    web_search: bool = False


@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(keep_alive())
    refresh_availability()
    print("=" * 60)
    print("AI ALL-IN-ONE v3")
    for m in MODELS:
        print(("  ✅ " if m["available"] else "  ❌ ") + m["name"])
    print("  🔍 بحث ويب: " + ("مفعّل" if TAVILY_KEY.strip() else "ناقص Tavily"))
    print("  افتح: http://localhost:" + str(PORT))
    print("=" * 60)
    yield


app = FastAPI(title="AI ALL-IN-ONE", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "models": MODELS,
        "has_any_key": any(m["available"] for m in MODELS),
        "web_search": bool(TAVILY_KEY.strip()),
    }


@app.post("/v1/chat")
async def chat(req: Req):
    if not req.messages:
        raise HTTPException(400, "لا توجد رسائل")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    sources = []

    if req.web_search and TAVILY_KEY.strip():
        query = messages[-1]["content"]
        results = await tavily_search(query)
        if results:
            sources = [{"title": r["title"], "url": r["url"]} for r in results]
            context = format_search_context(results)
            messages = messages[:-1] + [
                {"role": "user", "content": context + "\n\nسؤال المستخدم: " + query}
            ]

    async def gen():
        if sources:
            yield "data: " + json.dumps({"type": "sources", "sources": sources}) + "\n\n"

        chain = fallback_chain(req.model)
        if not chain:
            yield "data: " + json.dumps({"type": "error", "message": "لا يوجد موديل متاح"}) + "\n\n"
            return

        last_error = None
        for m in chain:
            try:
                if m["id"] != req.model:
                    yield "data: " + json.dumps({
                        "type": "notice",
                        "message": f"⚠️ تم التبديل تلقائيًا إلى {m['name']}"
                    }) + "\n\n"

                got_any = False
                async for tok in stream_chat(messages, m["id"]):
                    got_any = True
                    yield "data: " + json.dumps({"type": "token", "text": tok}) + "\n\n"

                if got_any:
                    yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                    return
            except ProviderBusy as e:
                last_error = str(e)
                continue
            except Exception as e:
                last_error = str(e)
                continue

        yield "data: " + json.dumps({
            "type": "error",
            "message": "كل الموديلات مشغولة دلوقتي. حاول تاني بعد شوية. (" + str(last_error) + ")"
        }) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ============================================================
# 🎨 الواجهة
# ============================================================

HTML = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI ALL-IN-ONE</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI",Tahoma,sans-serif;background:#0d1117;color:#e6edf3;height:100vh;display:flex;flex-direction:column}
header{background:#161b22;padding:12px 20px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
h1{font-size:17px;color:#58a6ff}
.ctrl{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
select{background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:6px 12px;font-family:inherit;font-size:13px;cursor:pointer;max-width:280px}
#st{font-size:11px;color:#8b949e}
.toggle{display:flex;align-items:center;gap:6px;background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:13px;user-select:none}
.toggle input{cursor:pointer;accent-color:#58a6ff}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.m{max-width:85%;padding:12px 16px;border-radius:12px;line-height:1.7;white-space:pre-wrap;word-wrap:break-word;font-size:15px}
.u{background:#1f6feb;color:#fff;align-self:flex-start}
.a{background:#21262d;border:1px solid #30363d;align-self:flex-end}
.e{background:#4a1516;border:1px solid #f85149;color:#ffa198;align-self:center;padding:10px;border-radius:8px;font-size:13px;max-width:90%}
.n{background:#3a2f0f;border:1px solid #d29922;color:#f0c674;align-self:center;padding:8px 12px;border-radius:8px;font-size:12px}
.sources{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;align-self:flex-end;max-width:85%;font-size:12px}
.sources h4{color:#58a6ff;font-size:12px;margin-bottom:6px}
.sources a{color:#7ee2a8;text-decoration:none;display:block;margin:4px 0;word-break:break-all}
.sources a:hover{text-decoration:underline}
footer{background:#161b22;padding:12px 20px;border-top:1px solid #30363d}
.r{display:flex;gap:8px;align-items:flex-end}
textarea{flex:1;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:10px 14px;font-family:inherit;font-size:14px;resize:none;min-height:48px;max-height:200px;outline:none}
textarea:focus{border-color:#58a6ff}
button{background:#238636;color:#fff;border:1px solid #2ea043;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer;font-family:inherit}
button:hover{background:#2ea043}
button:disabled{opacity:.5;cursor:not-allowed}
</style>
</head>
<body>
<header>
  <h1>🧠 AI ALL-IN-ONE</h1>
  <div class="ctrl">
    <span id="st">جاري التحميل...</span>
    <label class="toggle" title="بحث ويب مباشر">
      <input type="checkbox" id="webSearch">
      <span>🔍 بحث ويب</span>
    </label>
    <select id="modelSel"></select>
  </div>
</header>
<div id="chat"></div>
<footer>
  <div class="r">
    <textarea id="in" placeholder="اكتب سؤالك هنا... (Enter يرسل)"></textarea>
    <button id="send">إرسال</button>
  </div>
</footer>
<script>
const chat=document.getElementById('chat'),
      inp=document.getElementById('in'),
      send=document.getElementById('send'),
      st=document.getElementById('st'),
      modelSel=document.getElementById('modelSel'),
      webSearch=document.getElementById('webSearch');
let history=[],busy=false;

async function loadStatus(){
  try{
    const r=await fetch('/healthz');
    const j=await r.json();
    modelSel.innerHTML='';
    let firstAvail=null;
    j.models.forEach(m=>{
      const o=document.createElement('option');
      o.value=m.id;
      o.textContent=m.name + (m.available ? '' : ' — مفتاح ناقص');
      o.disabled=!m.available;
      if(m.available && !firstAvail) firstAvail=m.id;
      modelSel.appendChild(o);
    });
    if(firstAvail) modelSel.value=firstAvail;
    const n=j.models.filter(m=>m.available).length;
    st.textContent = n ? ('✅ متاح: '+n+'/'+j.models.length) : '❌ مفيش مفاتيح';
    if(!j.web_search){
      webSearch.disabled=true;
      webSearch.parentElement.style.opacity='0.5';
      webSearch.parentElement.title='مفتاح Tavily ناقص';
    }
  }catch(e){st.textContent='خطأ اتصال';}
}

function add(c,t){
  const d=document.createElement('div');
  d.className='m '+c;d.textContent=t;
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
  return d;
}

function addSources(list){
  const d=document.createElement('div');
  d.className='sources';
  let html='<h4>📚 المصادر:</h4>';
  list.forEach(s=>{
    html += '<a href="'+s.url+'" target="_blank">• '+s.title+'</a>';
  });
  d.innerHTML=html;
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
}

async function go(){
  if(busy)return;
  const t=inp.value.trim();
  if(!t)return;
  if(!modelSel.value) return;
  busy=true;send.disabled=true;
  add('u',t);
  inp.value='';
  history.push({role:'user',content:t});
  let div=null;
  let full='';
  try{
    const r=await fetch('/v1/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        messages:history,
        model:modelSel.value,
        web_search:webSearch.checked
      })
    });
    if(!r.ok)throw new Error(await r.text());
    const reader=r.body.getReader(),dec=new TextDecoder();
    let acc='';
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      acc+=dec.decode(value,{stream:true});
      const lines=acc.split('\\n');
      acc=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        try{
          const j=JSON.parse(line.slice(6));
          if(j.type==='token'){
            if(!div) div=add('a','');
            full+=j.text;
            div.textContent=full;
            chat.scrollTop=chat.scrollHeight;
          } else if(j.type==='sources'){
            addSources(j.sources);
          } else if(j.type==='notice'){
            add('n',j.message);
            div=null;
            full='';
          } else if(j.type==='error'){
            if(div) div.remove();
            add('e','خطأ: '+j.message);
            history.pop();
            return;
          }
        }catch(e){}
      }
    }
    if(full)history.push({role:'assistant',content:full});
  }catch(e){
    if(div) div.remove();add('e','خطأ: '+e.message);history.pop();
  }finally{
    busy=false;send.disabled=false;inp.focus();
  }
}

send.onclick=go;
inp.onkeydown=(e)=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();go();}};
loadStatus();
add('a','أهلاً! 👋 لو موديل وقع، هيتبدل تلقائيًا.');
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")