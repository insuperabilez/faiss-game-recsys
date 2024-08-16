from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
from typing import List
import pandas as pd
import numpy as np
import faiss
import os
import uvicorn
import pickle


app = FastAPI()

vectors = pd.read_csv('vectors.csv')
vectors_array = np.vstack(vectors['vector'].apply(lambda x: np.fromstring(x, sep=' '))).astype('float32')
vectors_array = vectors_array / np.linalg.norm(vectors_array, axis=1)[:, np.newaxis]
dimension = vectors_array.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectors_array)
with open('id_to_name.pkl', 'rb') as f:
    id_to_name = pickle.load(f)
with open('name_to_id.pkl', 'rb') as f:
    name_to_id = pickle.load(f)
# Определение схемы для индекса
schema = Schema(title=TEXT(stored=True))

# Создание индекса, если он не существует
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
    ix = create_in("indexdir", schema)
    writer = ix.writer()
    with open("names.txt", "r", encoding="utf-8") as file:
        for line in file:
            writer.add_document(title=line.strip())
    writer.commit()
else:
    ix = open_dir("indexdir")

# Функция для получения рекомендаций
def get_recs(name: str) -> str:
    selected_appid = name_to_id[name]
    selected_vector = vectors[vectors['AppID'] == selected_appid]['vector'].values[0]
    selected_vector = np.fromstring(selected_vector, sep=' ').astype('float32')
    selected_vector = np.array([selected_vector / np.linalg.norm(selected_vector)])
    k = 10
    distances, indices = index.search(selected_vector, k)

    # Получение AppID ближайших соседей
    nearest_appids = vectors.iloc[indices[0]]['AppID'].values
    result = ''
    for i in nearest_appids[1:]:
                result += f'<div class=block>{id_to_name[i]}</div>'
    return result

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Поиск игры</title>
        <style>
            
            body { display: flex; justify-content: center; align-items: center; height: 100vh; flex-direction: column; }
            input { width: 300px; padding: 10px; margin-bottom: 0; }
            .block {border: 1px solid #ccc; padding: 10px; margin: 10px 0; border-radius: 5px;}
            #suggestions { 
                width: 300px; 
                height: 200px; 
                overflow-y: auto; 
                display: flex; 
                flex-direction: column; 
                padding: 0; 
                margin: 0; 
            }
            li { 
                padding: 10px; 
                cursor: pointer; 
                border: 1px solid #ccc; 
                margin: 0; 
                border-radius: 5px; 
            }
            li:hover { background-color: #f0f0f0; }
        </style>
    </head>
    <body>
        <input type="text" id="search" placeholder="Введите название игры" oninput="searchGames()">
        <ul id="suggestions"></ul>
        <div id="recommendations"></div>
        <script>
            let lastSuggestions = [];

            async function searchGames() {
                const query = document.getElementById('search').value;
                const response = await fetch(`/search?query=${query}`);
                const suggestions = await response.json();
                const suggestionsList = document.getElementById('suggestions');
                suggestionsList.innerHTML = '';

                if (suggestions.length > 0) {
                    lastSuggestions = suggestions;
                }

                lastSuggestions.forEach(game => {
                    const li = document.createElement('li');
                    li.textContent = game;
                    li.onclick = () => getRecommendations(game);
                    suggestionsList.appendChild(li);
                });
            }

            async function getRecommendations(game) {
                const response = await fetch(`/recommendations?name=${game}`);
                const recommendations = await response.text();
                document.getElementById('recommendations').innerHTML = recommendations;
            }
        </script>
    </body>
    </html>
    """

@app.get("/search")
async def search(query: str):
    with ix.searcher() as searcher:
        query_parser = QueryParser("title", ix.schema)
        myquery = query_parser.parse(query)
        results = searcher.search(myquery, limit=5)
        return [result['title'] for result in results]

@app.get("/recommendations")
async def recommendations(name: str):
    return get_recs(name)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
