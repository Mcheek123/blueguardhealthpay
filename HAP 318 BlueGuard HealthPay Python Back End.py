from flask import Flask, render_template_string
from faker import Faker
import logging

app = Flask(__name__)
fake = Faker()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    try:
        with open("C:/Users/Mchee/OneDrive/Desktop/Documents/HAP 318 BlueGuard HealthPay/BlueGuard-HealthPay-Fresh/templates/home.html") as f:
            home_html = f.read()
        fake_user = generate_fake_user_data()  # Generate fake user data
        return render_template_string(home_html, fake_user=fake_user)
    except Exception as e:
        app.logger.error(f"Error in index route: {e}")
        return str(e), 500

def generate_fake_user_data():
    try:
        user_data = {
            'name': fake.name(),
            'patient_id': fake.uuid4(),
            'email': fake.email(),
            'address': fake.address(),
            'credit_card': {
                'number': fake.credit_card_number(card_type='visa'),
                'expiry': fake.credit_card_expire(),
                'cvv': fake.credit_card_security_code()
            }
        }
        return user_data
    except Exception as e:
        app.logger.error(f"Error in generate_fake_user_data function: {e}")
        raise

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081, debug=True, use_reloader=False)
    except Exception as e:
        app.logger.error(f"Error running the Flask app: {e}")
        raise
