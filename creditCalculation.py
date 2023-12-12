# credit_calculation.py 파일
import matplotlib.pyplot as plt
import boto3
from datetime import datetime, timedelta
# S3 클라이언트 생성
s3 = boto3.client('s3')

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

def plot_credits(credits, instance_name,instance_type):
    # 시간당 CPU 사용률 데이터의 길이에 따라 x축 (시간)을 설정합니다.
    time_hours = list(range(len(credits)))

    # 누적 크레딧 밸런스 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(
        time_hours, credits, marker="o", color="#1f77b4", markersize=3
    ) 
    plt.title("Credit Balance visualization")
    plt.xlabel("Time (5min)")
    plt.ylabel("Credit Balance")
    
    image_file = f"{instance_name}_{instance_type}.png"

    # 그림을 파일로 저장
    plt.savefig(image_file)
    plt.close()  # 현재 figure를 종료

    # 그림을 S3에 업로드
    with open(image_file, "rb") as data:
        s3.upload_fileobj(data, 'eta-credit-balance-graph', f"{datetime.today().strftime('%Y-%m-%d')}/{image_file}",ExtraArgs={
                'Metadata': {
                    'Content-Disposition': 'inline'
                },
                'ContentType': 'image/png' 
            })

