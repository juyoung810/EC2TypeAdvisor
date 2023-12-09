import json

import matplotlib.pyplot as plt

# Load the data from a JSON file
with open("data.json", "r") as file:
    test_data = json.load(file)

cpu_usage_values = test_data["cpuUsage"]["Values"]
credit = test_data["creditBalance"]["Values"][1:]  # 버스트 타입인 경우 비교용

# Constants for calculation
vcpu_count = 2
baseline_utilization = 0.2


# 버스트 타입이 아닌 타입을 버스트 타입이라 가정하고 크레딧 계산
def calculate_credits(cpu_usage_values):
    credit_acquired = []
    for cpu_util in cpu_usage_values:
        # 5분동안 얻는 크레딧
        earned_credits = (baseline_utilization * vcpu_count * 60) / 12

        # 5분에 한번 얻은 cpu 사용량에 따른 크레딧 소비량
        consumed_credits = vcpu_count * cpu_util

        # 크레딧 잔액 계산
        if cpu_util > baseline_utilization:
            net_credits = earned_credits - consumed_credits
            if net_credits < 0:
                net_credits = 0
        else:
            net_credits = earned_credits

        credit_acquired.append(net_credits)

    return credit_acquired


# Calculate credits
credits = calculate_credits(cpu_usage_values)

# 시간당 CPU 사용률 데이터의 길이에 따라 x축 (시간)을 설정합니다.
time_hours = list(range(len(credits)))

# 누적 크레딧 밸런스 시각화
plt.figure(figsize=(10, 6))
plt.plot(time_hours, credits, marker="o", color="b")
# plt.plot(time_hours, credit, marker="o", color="r")
plt.title("Accumulated Credit Balance for m5.large Instance")
plt.xlabel("Time (Hours)")
plt.ylabel("Accumulated Credit Balance")
plt.grid(True)
plt.show()
