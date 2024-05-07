variable "key_name" {
  description = "The key name of the EC2 key pair"
  type        = string
}

variable "region" {
  description = "The AWS region to create resources in"
  default     = "us-west-2"
  type        = string
}

variable "project" {
  description = "The project name, used as a prefix for resource names"
  default     = "KAN-Former"
  type        = string
}

variable "sagemaker_role_arn" {
  description = "The ARN of the IAM role for SageMaker"
  type        = string
}
