import os
import json
import datetime
import requests

def lambda_handler(event, context):
    sns_message = json.loads(event['Records'][0]['Sns']['Message'])
    post_data = build_slack_message(sns_message)
    post_slack(post_data, os.environ['SLACK_WEBHOOK_URL'])

def build_slack_message(data):
    execute_time = to_yyyymmddhhmmss(data['StateChangeTime'])
    description = data['AlarmDescription']
    instance_id = data['Trigger']['Dimensions'][0]['value']

    region_map = {
        "US East (Ohio)": "us-east-2",
        "US East (N. Virginia)": "us-east-1",
        "US West (N. California)": "us-west-1",
        "US West (Oregon)": "us-west-2",
    }
    
    region_code = region_map.get(data['Region'], "")

    # AWS Management Console에서 해당 인스턴스 페이지로 연결하는 링크 생성
    instance_link = f"https://console.aws.amazon.com/ec2/v2/home?region={region_code}#InstanceDetails:instanceId={instance_id}"

    return {
        'attachments': [
            {
                'title': data['AlarmName'],
                'color': "danger",
                'fields': [
                    {
                        'title': '발생 일자',
                        'value': execute_time
                    },
                    {
                        'title': '발생 인스턴스',
                        'value': instance_id
                    },
                    {
                        'title': '원인 설명',
                        'value': description
                    },
                    {
                        'title': '바로가기',
                        'value': instance_link
                    }
                ]
            }
        ]
    }

def to_yyyymmddhhmmss(time_string):
    if not time_string:
        return ''

    kst_date = datetime.datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f%z") + datetime.timedelta(hours=9)
    return kst_date.strftime("%Y-%m-%d %H:%M:%S")
    
def post_slack(message, slack_url):
    response = requests.post(slack_url, headers={'Content-Type': 'application/json'}, data=json.dumps(message))
    return response.text
