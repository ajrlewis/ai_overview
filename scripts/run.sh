source .env
source venv/bin/activate
query=${1}
python3 src/ai_search/ai_search.py "${query}"