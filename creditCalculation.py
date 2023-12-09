# credit_calculation.py 파일
import matplotlib.pyplot as plt

# 버스트 타입이 아닌 타입을 버스트 타입이라 가정하고 크레딧 계산
def calculate_credits(cpu_usage_values, vcpu_count, baseline_utilization):
    credit_acquired = []
    for cpu_util in cpu_usage_values:
        # 5분동안 얻는 크레딧
        earned_credits = (baseline_utilization * vcpu_count * 60) / 12

        # 5분에 한번 얻은 cpu 사용량에 따른 크레딧 소비량
        consumed_credits = vcpu_count * (cpu_util / 100)

        # 크레딧 잔액 계산
        if cpu_util > baseline_utilization:
            net_credits = earned_credits - consumed_credits
            if net_credits < 1.0: # 1 미만될 경우, 다른 t type 사용해서 base 올려봄
                return None
        else:
            net_credits = earned_credits

        credit_acquired.append(net_credits)

    return credit_acquired

def plot_credits(credits, instance_name):
    # 시간당 CPU 사용률 데이터의 길이에 따라 x축 (시간)을 설정합니다.
    time_hours = list(range(len(credits)))

    # 누적 크레딧 밸런스 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(time_hours, credits, marker="o", color="b")
    plt.title(f"Accumulated Credit Balance for {instance_name} Instance")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Accumulated Credit Balance")
    plt.grid(True)
    #plt.show()
    plt.savefig('credits.png')  # 그림을 파일로 저장
    plt.close()  # 현재 figure를 종료
