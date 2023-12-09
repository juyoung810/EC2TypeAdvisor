import boto3
import csv
from datetime import datetime, timedelta

# CloudWatch client 생성
cloudwatch = boto3.client('cloudwatch')

# EC2 client 생성
ec2 = boto3.resource('ec2')

# 'eta-'로 시작하는 인스턴스 필터링
instances = ec2.instances.filter(Filters=[{'Name': 'tag:Name', 'Values': ['eta-*']}])

for instance in instances:
    instance_id = instance.id

    # 인스턴스 정보 가져오기
    instance = ec2.Instance(instance_id)

    # 인스턴스 이름과 타입 가져오기
    instance_name = ''
    for tag in instance.tags:
        if tag['Key'] == 'Name':
            instance_name = tag['Value']
    instance_type = instance.instance_type

    # 메트릭 리스트
    metrics = ['CPUUtilization', 'CPUCreditBalance','CPUCreditUsage', 'mem_used_percent']

    # 7일 전부터 현재까지의 기간 설정
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)

    # CSV 파일 이름 설정
    csv_file_name = '{}_{}_metrics.csv'.format(instance_name, instance_type)

    # CSV 파일로 저장
    with open(csv_file_name, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["time", "CPUUtilization", "CPUCreditBalance", "CPUCreditUsage", "mem_used_percent"])
        writer.writeheader()

        data = {}
        for metric in metrics:
            if metric in ['CPUUtilization', 'CPUCreditBalance', 'CPUCreditUsage']:
                namespace = 'AWS/EC2'
            else:
                namespace = 'CWAgent'
            response = cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric,
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average'],
            )

            for datapoint in response['Datapoints']:
                timestamp = datapoint['Timestamp']

                if metric in ['CPUUtilization', 'CPUCreditBalance', 'CPUCreditUsage', 'mem_used_percent']:
                    data.setdefault(timestamp, {})[metric] = datapoint['Average']

        # 시간을 최신순으로 정렬
        sorted_data = dict(sorted(data.items(), key=lambda item: item[0], reverse=True))

        for timestamp, metrics in sorted_data.items():
            writer.writerow({
                'time': timestamp,
                'CPUUtilization': metrics.get('CPUUtilization', ''),
                'CPUCreditBalance': metrics.get('CPUCreditBalance', ''),
                'CPUCreditUsage': metrics.get('CPUCreditUsage', ''),
                'mem_used_percent': metrics.get('mem_used_percent', '')
            })