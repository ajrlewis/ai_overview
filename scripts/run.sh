source .env
source venv/bin/activate
query=${1}
python3 src/ai_overview/ai_overview.py "${query}"