def calculate_stop_loss_rrr(entry_price, take_profit, risk_reward_ratio):
    # Calculate price distance to take profit
    price_distance_to_tp = take_profit - entry_price

    # Calculate required distance to stop loss based on RRR
    price_distance_to_sl = price_distance_to_tp / risk_reward_ratio

    # Calculate stop loss price
    stop_loss = entry_price - price_distance_to_sl

    # Calculate potential risk and reward in points
    risk_points = entry_price - stop_loss
    reward_points = take_profit - entry_price

    # Calculate actual RRR
    actual_rrr = reward_points / risk_points

    return {
        "entry_price": entry_price,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "risk_points": risk_points,
        "reward_points": reward_points,
        "risk_reward_ratio": actual_rrr
    }


# Input values
#entry = 87945.93
#tp = 67515.83
#rrr = 3

# Calculate
#result = calculate_stop_loss_rrr(entry, tp, rrr)

# Print results
#print(f"Entry Price: {result['entry_price']:.4f}")
#print(f"Take Profit: {result['take_profit']:.4f}")
#print(f"Stop Loss: {result['stop_loss']:.4f}")
#print(f"Risk (points): {result['risk_points']:.4f}")
#print(f"Reward (points): {result['reward_points']:.4f}")
#print(f"Risk:Reward Ratio: 1:{result['risk_reward_ratio']:.2f}")