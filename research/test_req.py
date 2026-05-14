import requests
res = requests.post('http://127.0.0.1:5000/shaming', json={'Version': '1.0', 'tokens': [{'text': 'Sí, me voy a hacer cargo de los gastos que surjan', 'path': 'foo'}]})
print(res.json())
