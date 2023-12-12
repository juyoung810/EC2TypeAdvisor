import boto3
from datetime import datetime, timedelta
import requests
import json
from creditCalculation import calculate_credits, plot_credits
import time
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

# Instance Data 5분 단위로 가져오기
def get_metric_data(instance,metric_name, namespace):
    response = cloudwatch.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=[{'Name': 'InstanceId', 'Value': instance.instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=['Average']
    )
    if response['Datapoints']:
        datapoints = response['Datapoints']
        datapoints.sort(key=lambda x: x['Timestamp'])
        return datapoints
    else:
        print(f"No data for {metric_name}")
        return []

# peak 횟수 & 지속 시간 
def count_peaks_and_duration(data, threshold, metric_name):
    peaks_start = []
    peaks_end = []
    in_peak = False
    for i in range(len(data[metric_name])):
        if data[metric_name][i]['Average'] > threshold and not in_peak:
            peaks_start.append(data[metric_name][i]['Timestamp'])
            in_peak = True
        elif data[metric_name][i]['Average'] <= threshold and in_peak:
            peaks_end.append(data[metric_name][i]['Timestamp'])
            in_peak = False

    if not peaks_end:
        print(f"{metric_name}에서 피크를 찾을 수 없습니다.")
        return 0, 0, 0, 0

    if len(peaks_start) > len(peaks_end):
        peaks_end.append(data[metric_name][-1]['Timestamp'])

    durations_in_seconds = [(end - start).total_seconds() for start, end in zip(peaks_start, peaks_end)]

    peak_cpu_utilization_means = []
    peak_cpu_utilization_max = []
    for start, end in zip(peaks_start, peaks_end):
        peak_data = [d for d in data[metric_name] if start <= d['Timestamp'] <= end]
        peak_mean = sum(d['Average'] for d in peak_data) / len(peak_data)
        peak_max = max(d['Average'] for d in peak_data)
        peak_cpu_utilization_means.append(peak_mean)
        peak_cpu_utilization_max.append(peak_max)

    return len(peaks_start), sum(durations_in_seconds) / len(durations_in_seconds), sum(peak_cpu_utilization_means) / len(peak_cpu_utilization_means), max(peak_cpu_utilization_max)


# Overprovisioning, UnderProvisioning, Optimized 상태 판단
def check_provisioning_status(data,instance_type, instance_name):
    cpu_avg = sum(d['Average'] for d in data['CPUUtilization']) / len(data['CPUUtilization'])

    try:
        memory_avg = sum(d['Average'] for d in data['mem_used_percent']) / len(data['mem_used_percent']) if len(data['mem_used_percent']) > 0 else 0
    except (KeyError, IndexError):
        memory_avg = 25


    # 필요한 vCPU 수 
    target_cpu_usage = 50
    required_vcpus = 2 * (cpu_avg / target_cpu_usage)

    if instance_type.startswith('t'):
        credit_avg = sum(d['Average'] for d in data['CPUCreditBalance']) / len(data['CPUCreditBalance'])

    recommand_type = None
    peak_num ,avg_duration,avg_cpu_usage_peak,max_cpu_usage_peak = count_peaks_and_duration(data, 40, 'CPUUtilization')


    
    cpu_usage_values = [d['Average'] for d in data['CPUUtilization']]
    timestamps = [d['Timestamp'] for d in data['CPUUtilization']]
    
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
        credit_graph_name = plot_credits(credits, timestamps, instance_name, instance_type)
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

    server_info = {
        'name': instance_name,
        'type' : instance_type,
        'cpu': round(cpu_avg, 3),
        'memory': round(memory_avg, 3),
        'credit': round(credit_avg, 3) if instance_type.startswith('t') else "N/A",
        'peak_num': peak_num,
        'avg_duration': round(avg_duration, 3) if instance_type.startswith('t') else "N/A",
        'avg_cpu_usage_peak': round(avg_cpu_usage_peak, 3) if instance_type.startswith('t') else "N/A",
        'max_cpu_usage_peak': round(max_cpu_usage_peak, 3) if instance_type.startswith('t') else "N/A",
        'recommended_type': recommand_type,
        'current_cost': round(current_daily_cost, 3),
        'recommended_cost': round(recommand_daily_cost, 3),
        'credit_graph_name': credit_graph_name
    } 
    return server_info
    

def recommend_instance_type(required_vcpus):
    instance_candidates = []
    instance_types_to_consider = ['m', 'c']

    for instance_type in instance_types_to_consider:
        instance_candidates.append(min((i for i in instance_types if i['name'].startswith(instance_type) and i['vCPU'] >= required_vcpus), key=lambda x: x['price']))
    
    selected_instance = min(instance_candidates, key=lambda x: x['price'])
    return selected_instance['name'], selected_instance['price']


def send_to_slack(server):
    webhook_url = os.environ['SLACK_WEBHOOK_URL']

    image_url = f"https://eta-credit-balance-graph.s3.us-east-2.amazonaws.com/{datetime.today().strftime('%Y-%m-%d')}/{server['credit_graph_name']}"


    # 메시지 생성
    message = f"안녕하세요! :wave: \n *{server['type']} type의 {server['name']}* 서버의 성능 리포트를 전해드릴게요:mag_right:\n\n"
    message += ":cloud: *현재 CPU 사용량*\n"
    server_message = f" - 평균 CPU 사용률: {server['cpu']}%\n"
    if server['memory'] == 0:
        server_message += " - 메모리 에이전트를 설치해주세요!\n"
    else:
        server_message += f" - 평균 메모리 사용률: {server['memory']}%\n"


    if server['type'].startswith('t'):
        server_message += f" - CPU 크레딧 밸런스: {server['credit']}\n"

    if server['peak_num'] > 0:
        server_message += f":cloud: *T type Peak 분석*\n - 피크 수: {server['peak_num']}\n - 평균 피크 지속시간: {server['avg_duration']}s\n - 피크 기간 동안 CPU 평균 활용률: {server['avg_cpu_usage_peak']}%\n - 피크 기간 동안 CPU 최대 활용률: {server['max_cpu_usage_peak']}%\n"
    else:
        server_message += ":cloud: 피크가 존재하지 않습니다.\n"

    server_message += ":rocket: *사용량을 기반으로 EC2 type 을 추천합니다*\n"
    server_message += f" - 추천 서버 타입: {server['recommended_type']}\n - 현재 일일 비용: ${server['current_cost']}\n - 추천된 일일 비용: ${server['recommended_cost']}\n - 한 시간당 절감될 비용: ${server['current_cost'] - server['recommended_cost']}\n"
    
    server_image = {}
    if server['recommended_type'].startswith('t'):
        server_image["image_url"] = image_url
        server_image["text"] = f"\n*추천된 서버의 예상 CreditBalance 사용량 그래프*\n"

    message += server_message

    message += "\n잘못된 점이나 개선할 사항이 있다면 언제든지 알려주세요!"

    # Slack으로 메시지 전송
    slack_data = {
    'attachments': [
            {
                'fallback': 'Required plain-text summary of the attachment.',
                'color': '#FF9900',  # 색상 바의 색상을 설정합니다. 이는 HEX 코드를 사용합니다.
                'text': message  # 이 텍스트가 색상 바 옆에 표시됩니다.
            }
        ]
    }
    if server['recommended_type'].startswith('t'):
        slack_data["attachments"].append(server_image)
    response = requests.post(
        webhook_url, data=json.dumps(slack_data),
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:%s' % (response.status_code, response.text)
        )
        
        
def lambda_handler(event,handler):
    # 모든 인스턴스 가져오기
    ec2 = boto3.client('ec2')
    response = ec2.describe_instances()

    # 'eta-'로 시작하는 인스턴스 ID 목록 생성
    eta_instance_ids = []
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instance_id = instance["InstanceId"]
            for tag in instance["Tags"]:
                if tag["Key"] == "Name" and tag["Value"].startswith("eta-"):
                    eta_instance_ids.append(instance_id)

    for instance_id in eta_instance_ids:
        # 인스턴스 정보 가져오기
        ec2 = boto3.resource('ec2')
        instance = ec2.Instance(instance_id)
        instance_type = instance.instance_type
        instance_name = ""
        for tag in instance.tags:
            if tag['Key'] == 'Name':
                instance_name = tag['Value']
                break
    
        # cloudwatch 데이터 요청
        data = {}
        for metric in metrics:
            if metric in ['CPUUtilization', 'CPUCreditBalance', 'CPUCreditUsage']:
                namespace = 'AWS/EC2'
            else:
                namespace = 'CWAgent'
            data[metric] = get_metric_data(instance,metric, namespace)
            
        server_info = check_provisioning_status(data,instance_type, instance_name)
        if server_info is not None:
            send_to_slack(server_info)
            time.sleep(1)
        