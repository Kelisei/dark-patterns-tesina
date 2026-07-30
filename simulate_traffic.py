import asyncio
import aiohttp
import pandas as pd
import time
import os

async def main():
    df = pd.read_csv('research/datasets/unified_dataset.csv')
    texts = df['content'].tolist()
    
    if os.path.exists("data/active_learning_log.csv"):
        os.remove("data/active_learning_log.csv")
    
    async with aiohttp.ClientSession() as session:
        # Hit ping to warm up
        await session.get("http://127.0.0.1:5000/ping")
        
        start = time.time()
        
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {"texts": [{"text": t, "id": str(j), "path": "/"} for j, t in enumerate(batch)]}
            async with session.post("http://127.0.0.1:5000/detect", json=payload) as resp:
                await resp.read()
                
        end = time.time()
        
    total_texts = len(texts)
    
    logged_count = 0
    if os.path.exists("data/active_learning_log.csv"):
        log_df = pd.read_csv("data/active_learning_log.csv")
        logged_count = len(log_df)
        
    print(f"Total Requests Processed: {total_texts}")
    print(f"Total Logged for Active Learning: {logged_count} ({(logged_count/total_texts)*100:.2f}%)")
    print(f"Time Taken: {end-start:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
