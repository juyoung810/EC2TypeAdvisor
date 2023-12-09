import boto3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# CloudWatch 클라이언트 생성
cloudwatch = boto3.client('cloudwatch')

# 지난 7일 동안의 데이터를 가져오기 위한 시작 시간과 종료 시간 설정
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=1)

metrics = ['CPUUtilization', 'CPUCreditBalance', 'CPUCreditUsage', 'mem_used_percent']

# Instance ID
TBurst30 = 'i-0a9b8f7f0264a8f2b'
TBurst60 = 'i-02f3ef08150c73e1e'
MBurst60 = 'i-0d61dff2661abad07'
MBurst30 = 'i-09988a775d697dcde'
CHigh = 'i-0fc716d3ce6f6c16e'
instance_id = TBurst30

 # 인스턴스 정보 가져오기
ec2 = boto3.resource('ec2')
instance = ec2.Instance(instance_id)
instance_type = instance.instance_type

print("####################################")
print(instance_type)
print("####################################")

def get_metric_data(metric_name, namespace):
    response = cloudwatch.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=['Average']
    )
    if response['Datapoints']:
        df = pd.DataFrame(response['Datapoints'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp')
        df = df.sort_index()
        return df
    else:
        print(f"No data for {metric_name}")
        return pd.DataFrame()
    return pd.DataFrame(response['Datapoints']).set_index('Timestamp').sort_index()

data = {}
for metric in metrics:
    if metric in ['CPUUtilization', 'CPUCreditBalance', 'CPUCreditUsage']:
        namespace = 'AWS/EC2'
    else:
        namespace = 'CWAgent'
    data[metric] = get_metric_data(metric, namespace)

import numpy as np

def count_peaks_and_duration(data, threshold, metric_name):
    over_threshold = data[metric_name]['Average'] > threshold
    changes = over_threshold.ne(over_threshold.shift())
    peaks_start = data[metric_name][changes & over_threshold].index
    peaks_end = data[metric_name][changes & ~over_threshold].index

    if len(peaks_start) == 0 or len(peaks_end) == 0:
        print(f"{metric_name}에서 피크를 찾을 수 없습니다.")
        return

    if peaks_start[0] > peaks_end[0]:  # 첫 번째 피크 종료 시간이 첫 번째 피크 시작 시간보다 먼저인 경우
        peaks_end = peaks_end[1:]

    if len(peaks_start) > len(peaks_end):  # 마지막 피크가 종료되지 않은 경우
        peaks_end = peaks_end.append(pd.Index([data[metric_name].index[-1]]))

    durations = peaks_end - peaks_start
    durations_in_seconds = np.array([duration.total_seconds() for duration in durations])

    # 피크 기간 동안의 CPU 활용률 평균 계산
    peak_cpu_utilization_means = []
    peak_cpu_utilization_max = []
    for start, end in zip(peaks_start, peaks_end):
        peak_data = data[metric_name][(data[metric_name].index >= start) & (data[metric_name].index <= end)]
        peak_cpu_utilization_means.append(peak_data['Average'].mean())
        peak_cpu_utilization_max.append(peak_data['Average'].max())

    print(f"{metric_name}에서 피크 수: {len(peaks_start)}")
    print(f"{metric_name}에서 각 피크의 평균 지속 시간: {durations_in_seconds.mean()} 초")
    print(f"{metric_name}에서 피크 기간 동안의 CPU 활용률 평균: {np.mean(peak_cpu_utilization_means)}")
    print(f"{metric_name}에서 피크 기간 동안의 CPU 활용률 최대값: {max(peak_cpu_utilization_max)}")

count_peaks_and_duration(data, 40, 'CPUUtilization')




def count_zero_credit_balance_periods(data, metric_name):
    zero_balance = (data[metric_name]['Average'] == 0)
    changes = zero_balance.ne(zero_balance.shift())
    zero_start = data[metric_name][changes & zero_balance].index
    non_zero_after_zero = data[metric_name][changes & ~zero_balance].index

    if len(zero_start) == 0:
        print(f"{metric_name}에서 0 잔액 기간을 찾을 수 없습니다.")
        return

    if len(non_zero_after_zero) > 0 and zero_start[0] > non_zero_after_zero[0]:  # 첫 번째 0 잔액 종료 시간이 첫 번째 0 잔액 시작 시간보다 먼저인 경우
        non_zero_after_zero = non_zero_after_zero[1:]

    if len(zero_start) > len(non_zero_after_zero):  # 마지막 0 잔액 기간이 종료되지 않은 경우
        non_zero_after_zero = non_zero_after_zero.append(pd.Index([pd.Timestamp.now(tz='UTC')]))

    durations = non_zero_after_zero - zero_start
    durations_in_seconds = np.array([duration.total_seconds() for duration in durations])

    print(f"{metric_name}에서 0 잔액 기간 수: {len(zero_start)}")
    for i, duration in enumerate(durations_in_seconds, start=1):
        print(f"0 잔액 기간 {i}: {duration} 초")

# 인스턴스 타입이 't'로 시작하는 경우에만 함수 호출
if instance_type.startswith('t'):
    count_zero_credit_balance_periods(data, 'CPUCreditBalance')

# Overprovisioning, UnderProvisioning, Optimized 상태 판단
def check_provisioning_status(data):
    cpu_avg = data['CPUUtilization']['Average'].mean()
    memory_avg = data['mem_used_percent']['Average'].mean()
    if instance_type.startswith('t'):
        credit_avg = data['CPUCreditBalance']['Average'].mean()
    else:
        credit_avg = 100
    print('Average CPU Utilization:', cpu_avg)
    print('Average Memory Used Percent:', memory_avg)
    print('Average CPU Credit Balance:', credit_avg)

    if cpu_avg < 30 and memory_avg < 30: # CPU와 메모리 사용률이 각각 30% 미만인 경우
        return 'Overprovisioning'
    elif cpu_avg > 70 or memory_avg > 70 or credit_avg < 30: # CPU 또는 메모리 사용률이 70% 초과, 또는 CPU Credit Balance가 30 미만인 경우
        return 'UnderProvisioning'
    else:
        return 'Optimized'

status = check_provisioning_status(data)
print('Provisioning Status:', status)