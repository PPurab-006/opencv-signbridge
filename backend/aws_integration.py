import boto3
import os
import json
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSIntegration:
    def __init__(self):
        self.bucket_name = os.getenv("AWS_S3_BUCKET", "signbridge-evaluation-artifacts")
        # Initialize only if credentials are somewhat present or we are forcing it for local test
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            try:
                self.s3_client = boto3.client('s3')
                self.enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize AWS S3 client: {e}. Running in local mode.")
                self.enabled = False
        else:
            logger.warning("AWS credentials not found in environment. Running in local fallback mode.")
            self.enabled = False

    def upload_evaluation_metrics(self, file_path="data/metrics.json"):
        """
        Upload evaluation artifacts to S3 if enabled.
        Otherwise, writes a local log confirming fallback behavior.
        """
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found.")
            return False

        if not self.enabled:
            logger.info("AWS integration disabled or not configured. Saving artifacts locally as fallback.")
            fallback_path = file_path + ".fallback_log"
            with open(fallback_path, "w") as f:
                f.write("Local fallback saved. AWS S3 upload bypassed.")
            return True

        object_name = os.path.basename(file_path)
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            logger.info(f"Successfully uploaded {file_path} to s3://{self.bucket_name}/{object_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected AWS error: {e}")
            return False
