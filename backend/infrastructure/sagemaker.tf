resource "aws_sagemaker_notebook_instance" "kan_former_notebook" {
  name          = "KAN-Former-Notebook"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = "ml.t2.medium"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "KAN-Former SageMaker Notebook"
  }
}

resource "aws_sagemaker_model" "kan_former_model" {
  name          = "KAN-Former-Model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image     = "${aws_ecr_repository.kan_former_repository.repository_url}:latest"
    model_data_url = "${aws_s3_bucket.model_artifacts.bucket_regional_domain_name}/${aws_s3_bucket_object.model_artifact.key}"
  }

  tags = {
    Purpose = "Anomaly Detection"
  }
}

resource "aws_sagemaker_endpoint_configuration" "kan_former_endpoint_config" {
  name = "KAN-Former-Endpoint-Config"

  production_variants {
    variant_name          = "DefaultVariant"
    model_name            = aws_sagemaker_model.kan_former_model.name
    initial_instance_count = 1
    instance_type         = "ml.m4.xlarge"
  }
}

resource "aws_sagemaker_endpoint" "kan_former_endpoint" {
  name                 = "KAN-Former-Endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.kan_former_endpoint_config.name

  tags = {
    Environment = "production"
  }
}
