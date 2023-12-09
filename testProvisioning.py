import boto3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from creditCalculation import calculate_credits, plot_credits

# instance type 종류
instance_types = [
    {"name": "t3.nano", "price": 0.0052, "vCPU": 2, "base_util": 0.05 ,"memory": "0.5GiB"},
    {"name": "t3.micro", "price": 0.0104, "vCPU": 2,"base_util": 0.1, "memory": "1GiB"},
    {"name": "t3.small", "price": 0.0208, "vCPU": 2,"base_util": 0.2, "memory": "2GiB"},
    {"name": "t3.medium", "price": 0.0416, "vCPU": 2, "base_util": 0.2,"memory": "4GiB"},
    {"name": "t3.large", "price": 0.0832, "vCPU": 2,"base_util": 0.3, "memory": "8GiB"},
    {"name": "t3.xlarge", "price": 0.1664, "vCPU": 4,"base_util": 0.3, "memory": "16GiB"},
    {"name": "t3.2xlarge", "price": 0.3328, "vCPU": 8, "base_util": 0.4,"memory": "32GiB"},
    {"name": "m4.large", "price": 0.10, "vCPU": 2, "memory": "8GiB"},
    {"name": "m4.xlarge", "price": 0.20, "vCPU": 4, "memory": "16GiB"},
    {"name": "m4.2xlarge", "price": 0.40, "vCPU": 8, "memory": "32GiB"},
    {"name": "m4.4xlarge", "price": 0.80, "vCPU": 16, "memory": "64GiB"},
    {"name": "m4.10xlarge", "price": 2.00, "vCPU": 40, "memory": "160GiB"},
    {"name": "m4.16xlarge", "price": 3.20, "vCPU": 64, "memory": "256GiB"},
    {"name": "m5.large", "price": 0.096, "vCPU": 2, "memory": "8GiB"},
    {"name": "m5.xlarge", "price": 0.192, "vCPU": 4, "memory": "16GiB"},
    {"name": "m5.2xlarge", "price": 0.384, "vCPU": 8, "memory": "32GiB"},
    {"name": "m5.4xlarge", "price": 0.768, "vCPU": 16, "memory": "64GiB"},
    {"name": "m5.8xlarge", "price": 1.536, "vCPU": 32, "memory": "128GiB"},
    {"name": "m5.12xlarge", "price": 2.304, "vCPU": 48, "memory": "192GiB"},
    {"name": "m5.16xlarge", "price": 3.072, "vCPU": 64, "memory": "256GiB"},
    {"name": "m5.24xlarge", "price": 4.608, "vCPU": 96, "memory": "384GiB"},
    {"name": "m5.metal", "price": 4.608, "vCPU": 96, "memory": "384GiB"},
    {"name": "c4.large", "price": 0.10, "vCPU": 2, "memory": "3.75GiB"}, 
    {"name": "c4.xlarge", "price": 0.199, "vCPU": 4, "memory": "7.5GiB"},
    {"name": "c4.2xlarge", "price": 0.398, "vCPU": 8, "memory": "15GiB"},
    {"name": "c4.4xlarge", "price": 0.796, "vCPU": 16, "memory": "30GiB"},
    {"name": "c4.8xlarge", "price": 1.591, "vCPU": 36, "memory": "60GiB"},
    {"name": "c5.large", "price": 0.085, "vCPU": 2, "memory": "4GiB"},
    {"name": "c5.xlarge", "price": 0.17, "vCPU": 4, "memory": "8GiB"},
    {"name": "c5.2xlarge", "price": 0.34, "vCPU": 8, "memory": "16GiB"},
    {"name": "c5.4xlarge", "price": 0.68, "vCPU": 16, "memory": "32GiB"},
    {"name": "c5.9xlarge", "price": 1.53, "vCPU": 36, "memory": "72GiB"},
    {"name": "c5.12xlarge", "price": 2.04, "vCPU": 48, "memory": "96GiB"},
    {"name": "c5.18xlarge", "price": 3.06, "vCPU": 72, "memory": "144GiB"},
    {"name": "c5.24xlarge", "price": 4.08, "vCPU": 96, "memory": "192GiB"},
    {"name": "c5.metal", "price": 4.08, "vCPU": 96, "memory": "192GiB"},
]


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
THigh = 'i-0831c9d3766acca22'
MLow = 'i-0d6e4de2088feae6e'

# 테스트 Instance 선택
instance_id = MLow

# 인스턴스 정보 가져오기
ec2 = boto3.resource('ec2')
instance = ec2.Instance(instance_id)
instance_type = instance.instance_type
instance_name = ""
for tag in instance.tags:
    if tag['Key'] == 'Name':
        instance_name = tag['Value']
        break


print("####################################")
print(instance_type)
print("####################################")

# Instance Data 5분 단위로 가져오기
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


# peak 횟수 & 지속 시간 
def count_peaks_and_duration(data, threshold, metric_name):
    over_threshold = data[metric_name]['Average'] > threshold
    changes = over_threshold.ne(over_threshold.shift())
    peaks_start = data[metric_name][changes & over_threshold].index
    peaks_end = data[metric_name][changes & ~over_threshold].index

    if len(peaks_start) == 0 or len(peaks_end) == 0:
        print(f"{metric_name}에서 피크를 찾을 수 없습니다.")
        return 0

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
    
    return len(peaks_start) # peak 갯수 return 


# 인스턴스 성능에 맞는 t type 가져오기 (not used)
def get_vcpu_and_baseline(instance):
     # 현재 인스턴스의 정보를 가져옴
    current_instance = None
    for instance in instance_types:
        if instance['name'] == instance_type:
            current_instance = instance
            break

    # 현재 인스턴스의 정보를 찾지 못한 경우
    if current_instance is None:
        return None, None

    # 't'로 시작하는 인스턴스 중에서 vCPU 수가 같은 인스턴스를 찾음
    for instance in instance_types:
        if instance['name'].startswith('t') and instance['vCPU'] == current_instance['vCPU']:
            return instance['vCPU'], instance['base_util']

    return None, None

# Overprovisioning, UnderProvisioning, Optimized 상태 판단
def check_provisioning_status(data):
    cpu_avg = data['CPUUtilization']['Average'].mean()
    memory_avg = data['mem_used_percent']['Average'].mean()
    print('Average CPU Utilization:', cpu_avg)
    print('Average Memory Used Percent:', memory_avg)
    # 필요한 vCPU 수 
    target_cpu_usage = 50
    required_vcpus = 2 * (cpu_avg / target_cpu_usage)

    if instance_type.startswith('t'):
        credit_avg = data['CPUCreditBalance']['Average'].mean()
        print('Average CPU Credit Balance:', credit_avg)

    recommand_type = None
    peak_num = count_peaks_and_duration(data, 40, 'CPUUtilization')


    
    cpu_usage_values = data['CPUUtilization']['Average'].tolist()

    # t type 이 다른 type 보다 저렴하므로 항상 고려
    # 't'로 시작하는 인스턴스 타입들을 순회
    for instance in filter(lambda i: i['name'].startswith('t'), instance_types):
        vcpu_count, baseline_utilization = instance['vCPU'], instance['base_util']
        
        # 필요한 CPU를 충족시키지 못하면 다음 't' 타입 인스턴스로 넘어감
        if vcpu_count < required_vcpus:
            continue
        # 크레딧 계산
        credits = calculate_credits(cpu_usage_values, vcpu_count, baseline_utilization)

        # 크레딧이 0 이하면 다음 인스턴스 타입으로 넘어감
        if credits is None:
            continue
        
        # 크레딧 시각화
        t_recommand_type = instance['name']
        plot_credits(credits, instance_name)
        break

     # 't' type 인스턴스 가격
    t_price = next(i for i in instance_types if i['name'] == t_recommand_type)['price']

    # 'm' 또는 'c' 타입 인스턴스 추천 받기
    recommand_type, recommand_price = recommend_instance_type(required_vcpus)

    # 't' type 인스턴스가 더 비싸면 'm' 또는 'c' 타입 인스턴스를 추천
    if t_price > recommand_price:
        recommand_type = recommand_type
    else:
        recommand_type = t_recommand_type

     # 현재 인스턴스 타입의 가격 (시간당)
    current_price = next(i for i in instance_types if i['name'] == instance_type)['price']
     # 추천 인스턴스 타입의 가격 (시간당)
    recommand_price = next(i for i in instance_types if i['name'] == recommand_type)['price']

    # 현재 인스턴스 타입과 추천 인스턴스 타입의 하루 비용 계산 및 출력
    current_daily_cost = current_price * 24
    recommand_daily_cost = recommand_price * 24
    print('--------------------------------')
    print('Recommend Type:', recommand_type)
    print(f'Current daily cost: ${current_daily_cost}')
    print(f'Recommended daily cost: ${recommand_daily_cost}')
    return
    

def recommend_instance_type(required_vcpus):
    instance_candidates = []
    instance_types_to_consider = ['m', 'c']

    for instance_type in instance_types_to_consider:
        instance_candidates.append(min((i for i in instance_types if i['name'].startswith(instance_type) and i['vCPU'] >= required_vcpus), key=lambda x: x['price']))
    
    selected_instance = min(instance_candidates, key=lambda x: x['price'])
    return selected_instance['name'], selected_instance['price']


status = check_provisioning_status(data)

