from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
import os
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')
CORS(app)

# Database setup
DB_PATH = 'keno_data.db'

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS draws
                 (id INTEGER PRIMARY KEY, draw_number INTEGER, numbers TEXT, date TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Global data storage
draws_data = []
analysis_cache = {}

def parse_upload_file(file):
    """Parse CSV or JSON file with Keno draws"""
    try:
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.json'):
            df = pd.read_json(file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return None, "Unsupported file format. Use CSV, JSON, or XLSX."
        
        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Extract numbers (expect columns like: number1, number2, etc. or 'numbers' as list)
        draws = []
        for idx, row in df.iterrows():
            numbers = []
            
            # Try to extract numbers from individual columns
            for col in df.columns:
                if 'number' in col or 'num' in col:
                    try:
                        num = int(row[col])
                        if 1 <= num <= 80:  # Standard Keno range
                            numbers.append(num)
                    except:
                        pass
            
            # If no numbers found, try 'numbers' column as list
            if not numbers and 'numbers' in df.columns:
                try:
                    if isinstance(row['numbers'], (list, str)):
                        nums = ast.literal_eval(row['numbers']) if isinstance(row['numbers'], str) else row['numbers']
                        numbers = [int(n) for n in nums if 1 <= int(n) <= 80]
                except:
                    pass
            
            if numbers:
                draw_date = None
                if 'date' in df.columns:
                    try:
                        draw_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                    except:
                        draw_date = datetime.now().strftime('%Y-%m-%d')
                else:
                    draw_date = datetime.now().strftime('%Y-%m-%d')
                
                draws.append({
                    'draw_number': idx + 1,
                    'numbers': sorted(list(set(numbers[:6]))),  # Take first 6 unique numbers
                    'date': draw_date
                })
        
        if not draws:
            return None, "No valid draws found in file."
        
        return draws, None
    
    except Exception as e:
        return None, str(e)

def save_draws_to_db(draws):
    """Save draws to SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for draw in draws:
        c.execute('INSERT INTO draws (draw_number, numbers, date) VALUES (?, ?, ?)',
                  (draw['draw_number'], json.dumps(draw['numbers']), draw['date']))
    conn.commit()
    conn.close()

def load_draws_from_db():
    """Load draws from SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT draw_number, numbers, date FROM draws')
    rows = c.fetchall()
    conn.close()
    
    draws = []
    for row in rows:
        draws.append({
            'draw_number': row[0],
            'numbers': json.loads(row[1]),
            'date': row[2]
        })
    return draws

def calculate_statistics(draws):
    """Calculate comprehensive statistics from draws"""
    if not draws:
        return {}
    
    # Frequency analysis
    all_numbers = []
    for draw in draws:
        all_numbers.extend(draw['numbers'])
    
    frequency = Counter(all_numbers)
    total_occurrences = len(all_numbers)
    
    # Hot and cold numbers
    hot_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    cold_numbers = [(n, frequency.get(n, 0)) for n in range(1, 81) if frequency.get(n, 0) > 0]
    cold_numbers = sorted(cold_numbers, key=lambda x: x[1])[:10]
    
    # Overdue analysis (numbers not in recent draws)
    recent_draws = draws[-20:] if len(draws) > 20 else draws
    recent_numbers = set()
    for draw in recent_draws:
        recent_numbers.update(draw['numbers'])
    
    overdue = [(n, frequency.get(n, 0)) for n in range(1, 81) if n not in recent_numbers and frequency.get(n, 0) > 0]
    overdue = sorted(overdue, key=lambda x: x[1], reverse=True)[:10]
    
    # Pair frequencies
    pairs = defaultdict(int)
    for draw in draws:
        nums = sorted(draw['numbers'])
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pair = (nums[i], nums[j])
                pairs[pair] += 1
    
    top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Triplet frequencies
    triplets = defaultdict(int)
    for draw in draws:
        nums = sorted(draw['numbers'])
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                for k in range(j+1, len(nums)):
                    triplet = (nums[i], nums[j], nums[k])
                    triplets[triplet] += 1
    
    top_triplets = sorted(triplets.items(), key=lambda x: x[1], reverse=True)[:10]
    
    stats = {
        'total_draws': len(draws),
        'total_unique_numbers_drawn': len(frequency),
        'frequency': {str(k): v for k, v in sorted(frequency.items())},
        'hot_numbers': [{'number': n, 'frequency': f} for n, f in hot_numbers],
        'cold_numbers': [{'number': n, 'frequency': f} for n, f in cold_numbers],
        'overdue_numbers': [{'number': n, 'frequency': f} for n, f in overdue],
        'top_pairs': [{'pair': list(p), 'frequency': f} for p, f in top_pairs],
        'top_triplets': [{'triplet': list(t), 'frequency': f} for t, f in top_triplets],
        'average_numbers_per_draw': round(total_occurrences / len(draws), 2)
    }
    
    return stats

def generate_tickets(draws, num_tickets=10):
    """Generate AI-suggested tickets based on patterns"""
    stats = calculate_statistics(draws)
    tickets = []
    
    if not stats:
        return tickets
    
    # Ticket 1-3: Hot number tickets
    hot_nums = [n['number'] for n in stats['hot_numbers']]
    for i in range(3):
        ticket = hot_nums[i*2:(i+1)*2+4]
        if len(ticket) == 6:
            tickets.append({'numbers': ticket, 'strategy': 'Hot Numbers'})
    
    # Ticket 4-6: Cold number tickets
    cold_nums = [n['number'] for n in stats['cold_numbers']]
    for i in range(3):
        ticket = cold_nums[i*2:(i+1)*2+4]
        if len(ticket) == 6:
            tickets.append({'numbers': ticket, 'strategy': 'Cold Numbers'})
    
    # Ticket 7: Pair-based
    pair_numbers = set()
    for pair in stats['top_pairs'][:5]:
        pair_numbers.update(pair['pair'])
    if len(pair_numbers) >= 6:
        tickets.append({'numbers': sorted(list(pair_numbers))[:6], 'strategy': 'Top Pairs'})
    
    # Ticket 8: Random selection
    random_ticket = sorted(np.random.choice(range(1, 81), 6, replace=False).tolist())
    tickets.append({'numbers': random_ticket, 'strategy': 'Random'})
    
    # Ticket 9: Balanced distribution (spread across ranges)
    balanced = []
    for range_start in [1, 14, 27, 40, 53, 66]:
        balanced.append(np.random.randint(range_start, min(range_start+13, 81)))
    tickets.append({'numbers': sorted(balanced), 'strategy': 'Balanced Distribution'})
    
    # Ticket 10: Overdue focused
    overdue_nums = [n['number'] for n in stats['overdue_numbers']]
    if len(overdue_nums) >= 6:
        tickets.append({'numbers': overdue_nums[:6], 'strategy': 'Overdue Numbers'})
    else:
        tickets.append({'numbers': sorted(np.random.choice(range(1, 81), 6, replace=False).tolist()), 'strategy': 'Random Fill'})
    
    return tickets[:10]

def backtest_strategy(draws):
    """Compare AI-generated tickets against random selections"""
    if len(draws) < 20:
        return {'error': 'Need at least 20 draws for backtesting'}
    
    # Generate tickets from first 80% of data
    train_size = int(len(draws) * 0.8)
    train_draws = draws[:train_size]
    test_draws = draws[train_size:]
    
    ai_tickets = generate_tickets(train_draws, num_tickets=10)
    
    # Score tickets against test draws
    ai_hits = 0
    random_hits = 0
    
    for test_draw in test_draws:
        test_numbers = set(test_draw['numbers'])
        
        # Check AI tickets
        for ticket in ai_tickets:
            matches = len(set(ticket['numbers']) & test_numbers)
            if matches >= 4:  # 4+ matches = hit
                ai_hits += 1
        
        # Check random tickets
        for _ in range(len(ai_tickets)):
            random_ticket = set(np.random.choice(range(1, 81), 6, replace=False))
            matches = len(random_ticket & test_numbers)
            if matches >= 4:
                random_hits += 1
    
    return {
        'total_test_draws': len(test_draws),
        'ai_strategy_hits': ai_hits,
        'random_strategy_hits': random_hits,
        'ai_hit_rate': round((ai_hits / (len(test_draws) * len(ai_tickets))) * 100, 2) if test_draws else 0,
        'random_hit_rate': round((random_hits / (len(test_draws) * len(ai_tickets))) * 100, 2) if test_draws else 0
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload():
    """Handle file upload"""
    global draws_data, analysis_cache
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    draws, error = parse_upload_file(file)
    if error:
        return jsonify({'error': error}), 400
    
    draws_data = draws
    save_draws_to_db(draws_data)
    analysis_cache = {}
    
    return jsonify({
        'success': True,
        'draws_loaded': len(draws_data),
        'message': f'Successfully loaded {len(draws_data)} draws'
    })

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get statistical analysis"""
    if not draws_data:
        return jsonify({'error': 'No data loaded'}), 400
    
    stats = calculate_statistics(draws_data)
    return jsonify(stats)

@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Generate AI tickets"""
    if not draws_data:
        return jsonify({'error': 'No data loaded'}), 400
    
    tickets = generate_tickets(draws_data, num_tickets=10)
    return jsonify({'tickets': tickets})

@app.route('/api/backtest', methods=['GET'])
def backtest():
    """Run backtest analysis"""
    if not draws_data:
        return jsonify({'error': 'No data loaded'}), 400
    
    results = backtest_strategy(draws_data)
    return jsonify(results)

@app.route('/api/status', methods=['GET'])
def status():
    """Get application status"""
    return jsonify({
        'draws_loaded': len(draws_data),
        'ready': len(draws_data) > 0
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
