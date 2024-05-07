module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "kan-former-cluster"
  cluster_version = "1.21"
  subnets         = module.vpc.private_subnets

  node_groups = {
    kan_former_nodes = {
      desired_capacity = 2
      max_capacity     = 3
      min_capacity     = 1

      instance_type = "m5.large"
      key_name      = var.key_name
    }
  }

  tags = {
    Project     = "KAN-Former"
    Environment = "Production"
  }
}
