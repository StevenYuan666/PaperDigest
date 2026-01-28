import os
import requests
import datetime
import arxiv
from openai import OpenAI

# --- 配置部分 ---
API_KEY = os.getenv("THIRD_PARTY_API_KEY") 
BASE_URL = "https://endpoint.greatrouter.com" 
MODEL_NAME = "gpt-5-nano"

# Discord 配置
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL") 

# 你的关键词
KEYWORDS = [
    "Agents", 
    "Large Language Models", 
    "Vision Language Models", 
    "LLM Personalization", 
    "RAG", 
    "Reasoning", 
    "Latent Reasoning"
]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 工具函数 ---

def get_arxiv_papers_by_keywords():
    """
    针对每个关键词单独搜索，不限制分类 (cs.CL, cs.LG, cs.CV, cs.AI 均可被搜到)。
    """
    print("正在根据关键词逐个抓取 Arxiv (Global Search)...")
    
    arxiv_client = arxiv.Client()
    collected_papers = []
    seen_ids = set() 

    for keyword in KEYWORDS:
        print(f"  > 正在搜索: {keyword} ...")
        
        # --- 核心修改 ---
        # 旧逻辑: f'cat:cs.CL AND (ti:"{keyword}" OR abs:"{keyword}")'
        # 新逻辑: 只要标题或摘要包含关键词即可，不限 Category
        query = f'ti:"{keyword}" OR abs:"{keyword}"'
        
        search = arxiv.Search(
            query=query,
            max_results=5, 
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        found_for_this_keyword = False
        
        for result in arxiv_client.results(search):
            # 1. 检查去重
            if result.entry_id in seen_ids:
                continue
            
            # 2. 检查时间（最近 48 小时）
            if (datetime.datetime.now(datetime.timezone.utc) - result.published).days > 2:
                continue
            
            # 3. (可选) 简单的噪声过滤
            # 虽然我们放开了分类，但为了防止 "Agents" 搜到纯经济学论文，
            # 可以检查一下 primary_category 是否属于计算机或统计学 (cs.*, stat.*)
            # 如果你想要最大范围，可以把下面这两行注释掉
            if not result.primary_category.startswith(('cs', 'stat')):
                 continue 

            collected_papers.append({
                "source": f"Arxiv [{result.primary_category}]", # 显示具体分类，方便你确认来源
                "title": result.title,
                "url": result.entry_id,
                "abstract": result.summary,
                "authors": ", ".join([a.name for a in result.authors[:3]]) + " et al.",
                "color": 16711680 # 红色
            })
            
            seen_ids.add(result.entry_id)
            found_for_this_keyword = True
            break # 找到一篇最新的就跳到下一个关键词
        
        if not found_for_this_keyword:
            print(f"    - {keyword}: 暂无最新相关论文")

    return collected_papers

def get_huggingface_daily_papers(max_results=2):
    """获取 Hugging Face 热门"""
    print("正在抓取 Hugging Face Daily Papers...")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    url = f"https://huggingface.co/api/daily_papers?date={today}"
    
    try:
        resp = requests.get(url)
        # 自动回退日期逻辑
        if resp.status_code != 200 or not resp.json():
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            url = f"https://huggingface.co/api/daily_papers?date={yesterday}"
            resp = requests.get(url)
            
        data = resp.json()
        if not isinstance(data, list): return []

        sorted_papers = sorted(data, key=lambda x: x.get('upvotes', 0), reverse=True)[:max_results]
        
        papers = []
        for p in sorted_papers:
            paper_info = p.get('paper', {})
            if not paper_info: continue
            
            papers.append({
                "source": "Hugging Face 🔥",
                "title": paper_info.get('title', 'Unknown'),
                "url": f"https://huggingface.co/papers/{paper_info.get('id', '')}",
                "abstract": paper_info.get('summary', ''), 
                "authors": "Community Trending", 
                "color": 16776960 # 黄色
            })
        return papers
    except Exception as e:
        print(f"HF抓取失败: {e}")
        return []

def summarize_with_ai(paper_data):
    """调用 API 总结"""
    prompt = f"""
    You are a research assistant. Summarize this paper for an expert.
    
    Paper Title: {paper_data['title']}
    Abstract: {paper_data['abstract']}
    
    Format output strictly:
    **TL;DR**: [One concise sentence]
    **Key Innovation**: [1-2 bullet points on technical novelty]
    **Performance**: [Main metric or result if available]
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Summary failed: {str(e)}"

def send_discord_embed(paper_data, summary):
    """发送 Discord Embed"""
    embed = {
        "title": paper_data['title'],
        "url": paper_data['url'],
        "description": summary,
        "color": paper_data['color'],
        "fields": [
            {
                "name": "Topic / Source",
                "value": paper_data['source'],
                "inline": True
            },
            {
                "name": "Authors",
                "value": paper_data['authors'],
                "inline": True
            }
        ],
        "footer": {
            "text": f"Generated by {MODEL_NAME} • {datetime.datetime.now().strftime('%Y-%m-%d')}"
        }
    }
    
    requests.post(WEBHOOK_URL, json={"embeds": [embed]})

# --- 主程序 ---
if __name__ == "__main__":
    all_papers = []
    
    # 1. 抓取 Arxiv (全库)
    try:
        all_papers.extend(get_arxiv_papers_by_keywords())
    except Exception as e:
        print(f"Arxiv 模块出错: {e}")
    
    # 2. 抓取 HF (热门补充)
    hf_papers = get_huggingface_daily_papers(max_results=3)
    
    # 去重
    existing_titles = {p['title'].lower() for p in all_papers}
    for hf in hf_papers:
        if hf['title'].lower() not in existing_titles:
            all_papers.append(hf)
    
    print(f"共获取到 {len(all_papers)} 篇论文，开始总结发送...")
    
    for paper in all_papers:
        print(f"处理: {paper['title']}")
        summary = summarize_with_ai(paper)
        send_discord_embed(paper, summary)
        
    print("全部完成！")