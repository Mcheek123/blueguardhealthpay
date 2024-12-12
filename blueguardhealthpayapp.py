from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process_payment', methods=['POST'])
def process_payment():
    transaction_data = request.json
    transaction_type = transaction_data.get('type')
    amount = transaction_data.get('amount')

    # Process the transaction based on its type
    if transaction_type == 'telehealth':
        return jsonify({'status': 'success', 'message': f'Telehealth payment of ${amount} processed'})
    elif transaction_type == 'insurance_copay':
        return jsonify({'status': 'success', 'message': f'Insurance co-pay of ${amount} processed'})
    elif transaction_type == 'parking':
        return jsonify({'status': 'success', 'message': f'Parking fee of ${amount} processed'})
    elif transaction_type == 'in_hospital_services':
        return jsonify({'status': 'success', 'message': f'In-hospital services payment of ${amount} processed'})
    elif transaction_type == 'pharmacy':
        return jsonify({'status': 'success', 'message': f'Pharmacy purchase of ${amount} processed'})
    elif transaction_type == 'diagnostic_tests':
        return jsonify({'status': 'success', 'message': f'Diagnostic test payment of ${amount} processed'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid transaction type'})

if __name__ == '__main__':
    app.run(debug=True)
