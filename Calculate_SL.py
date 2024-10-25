def calculate_stop_loss(entry_price, atr_value, atr_multiplier, trade_dir):
    if trade_dir == "B":
        stop_loss = entry_price - (atr_value * atr_multiplier)
    else:
        stop_loss = entry_price + (atr_value * atr_multiplier)

    return stop_loss

try:
    entry_price = float(input("Enter the entry price: "))
    atr_value = float(input("Enter the ATR value: "))
    atr_multiplier = float(input("Enter the ATR multiplier: "))
    trade_direction = str(input("Enter trade direction: Bullish(B), Bearish(H): "))

    # Calculate stop loss
    stop_loss = calculate_stop_loss(entry_price, atr_value, atr_multiplier, trade_direction)

    print(f"The calculated stop loss is: {stop_loss:.2f}")
except ValueError:
    print("Please enter valid numerical values.")