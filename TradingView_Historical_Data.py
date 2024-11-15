import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Configure Selenium options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run headless Chrome
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Path to your ChromeDriver
chrome_service = Service('D:/DOWNLOAD/chromedriver-win64/chromedriver-win64/chromedriver.exe')  # Update this path

# Start the WebDriver
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# Function to extract data from TradingView
def extract_tradingview_data(symbol):
    url = f'https://www.tradingview.com/symbols/{symbol}/'  # Modify as needed
    driver.get(url)
    time.sleep(5)  # Wait for page to load=

    # Example: Extracting the current price
    try:
        price_element = driver.find_element(By.CSS_SELECTOR, 'div.price')  # Update selector as needed
        price = price_element.text
        print(f"Current price of {symbol}: {price}")
    except Exception as e:
        print("Could not extract the data:", e)

# Example usage
extract_tradingview_data('AAPL')  # Replace 'AAPL' with the desired stock symbol

# Close the WebDriver
driver.quit()