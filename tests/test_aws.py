from backend.aws_integration import AWSIntegration
import os

def test_aws_integration_fallback():
    # Ensure no credentials
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    
    aws = AWSIntegration()
    assert aws.enabled == False
    
    # Test fallback write
    test_file = "test_metrics.json"
    with open(test_file, "w") as f:
        f.write("{}")
        
    result = aws.upload_evaluation_metrics(test_file)
    assert result == True
    assert os.path.exists(test_file + ".fallback_log")
    
    # Cleanup
    os.remove(test_file)
    os.remove(test_file + ".fallback_log")
