import logging
import boto3
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print('Loading Lambda function')

endpoint_name='autogluon-inference-2023-04-13-00-11-13-621'

def lambda_handler(event, context):

    print('Context:::',context)
    print('EventType::',type(event))
    
    # Extracting S3 info from URL
    s3_url = event['s3_url']
    s3_url = s3_url[5:]
    url_tokens = s3_url.split("/")
    bucket=url_tokens[0]
    prefix='/'.join(url_tokens[1:])
    file=url_tokens[-1]
    
    # CSV download from s3
    s3 = boto3.resource('s3')
    s3.meta.client.download_file(bucket, prefix, f"/tmp/{file}")
    
    df_test = pd.read_csv(f"/tmp/{file}")
    df_test_mod = df_test.drop(columns=["customer_ID", "S_2", "target"], axis=1)
    
    # Create CSV payload
    csv_file = StringIO()
    df_test_mod.to_csv(csv_file, sep=",", header=True, index=False)
    csv_payload = csv_file.getvalue()
    
    # Invoke Endpoint
    runtime=boto3.Session().client('sagemaker-runtime')
    response=runtime.invoke_endpoint(EndpointName=endpoint_name,
                                    ContentType="text/csv",
                                    Accept='text/csv',
                                    Body=csv_payload)
    
    # Converting the response body into Pandas Dataframe
    csv_str=response['Body'].read().decode('utf-8')
    df_result=pd.read_csv(StringIO(csv_str), sep=",", header=0)
    
    # Comparing predictions with actual values and storing the result
    predictions = df_result[["pred"]]
    predictions = predictions.join(df_test["target"]).rename(columns={"target": "actual"})
    result = f"{(predictions.pred==predictions.actual).astype(int).sum()}/{len(predictions)} are correct"
    
    return {
        'statusCode': 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'type-result':str(type(result)),
        'Content-Type':str(context),
        'body' : result
    }
