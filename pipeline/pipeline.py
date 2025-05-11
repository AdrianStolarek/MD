from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component
from typing import Dict
import os
import json

@component(
    base_image="{{image_uri}}", 
    output_component_file="train_component.yaml"
)
def train_model(
    model_dir: str,
) -> Dict[str, float]:

    import json
    import os
    
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    return metrics

@component(
    packages_to_install=["google-cloud-aiplatform"],
    output_component_file="register_component.yaml"
)
def register_model(
    model_display_name: str,
    model_path: str,
    project: str,
    region: str,
    metrics: Dict[str, float]
) -> str:

    from google.cloud import aiplatform
    
    aiplatform.init(project=project, location=region)
    
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_path,
        serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest",
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        instance_schema_uri=None,
        parameters_schema_uri=None,
        prediction_schema_uri=None,
        explanation_metadata=None,
        explanation_parameters=None,
        metadata={
            "accuracy": str(metrics["accuracy"])
        }
    )
    
    print(f"Model zarejestrowany: {model.resource_name}")
    return model.resource_name

@component(
    packages_to_install=["google-cloud-aiplatform"],
    output_component_file="deploy_component.yaml"
)
def deploy_model(
    model_resource_name: str,
    endpoint_name: str,
    project: str,
    region: str,
    machine_type: str = "n1-standard-2",
) -> str:

    from google.cloud import aiplatform
    
    aiplatform.init(project=project, location=region)
    
    model = aiplatform.Model(model_resource_name)
    
    # Sprawdź, czy endpoint istnieje
    try:
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"',
            order_by="create_time desc",
            project=project, 
            location=region
        )
        
        if endpoints:
            endpoint = endpoints[0]
            print(f"Znaleziono istniejący endpoint: {endpoint.display_name}")
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                project=project,
                location=region
            )
            print(f"Utworzono nowy endpoint: {endpoint.display_name}")
            
    except Exception as e:
        print(f"Błąd podczas sprawdzania endpointu: {e}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            project=project,
            location=region
        )
        print(f"Utworzono nowy endpoint: {endpoint.display_name}")
    
    deployment = endpoint.deploy(
        model=model,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=1
    )
    
    print(f"Model wdrożony: {endpoint.resource_name}")
    endpoint_url = f"https://{region}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict"
    return endpoint_url

@dsl.pipeline(
    name="pytorch-training-pipeline",
    description="Pipeline do trenowania i wdrażania modelu PyTorch w Vertex AI"
)
def pytorch_training_pipeline(
    project: str,
    region: str,
    gcs_bucket: str,
    model_display_name: str,
    endpoint_name: str
):
    output_dir = f"gs://{gcs_bucket}/models/{model_display_name}"
    
    train_task = train_model(
        model_dir=output_dir
    )
    
    register_task = register_model(
        model_display_name=model_display_name,
        model_path=output_dir,
        project=project,
        region=region,
        metrics=train_task.outputs["Output"]
    )
    register_task.after(train_task)
    
    deploy_task = deploy_model(
        model_resource_name=register_task.outputs["Output"],
        endpoint_name=endpoint_name,
        project=project,
        region=region
    )
    deploy_task.after(register_task)

def compile_pipeline(
    output_path: str,
    image_uri: str,
    project: str,
    region: str,
    gcs_bucket: str,
    model_display_name: str,
    endpoint_name: str
):

    pipeline_def = pytorch_training_pipeline
    pipeline_def.component_spec.implementation.container.image = image_uri
    
    compiler.Compiler().compile(
        pipeline_func=pipeline_def,
        package_path=output_path,
        pipeline_parameters={
            "project": project,
            "region": region,
            "gcs_bucket": gcs_bucket,
            "model_display_name": model_display_name,
            "endpoint_name": endpoint_name
        }
    )
    
    print(f"Pipeline skompilowany i zapisany w {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", required=True, help="Ścieżka do pliku wyjściowego JSON")
    parser.add_argument("--image-uri", required=True, help="URI obrazu Docker z modelem")
    parser.add_argument("--project", required=True, help="ID projektu Google Cloud")
    parser.add_argument("--region", required=True, help="Region Google Cloud")
    parser.add_argument("--gcs-bucket", required=True, help="Nazwa bucketu GCS")
    parser.add_argument("--model-display-name", required=True, help="Nazwa modelu")
    parser.add_argument("--endpoint-name", required=True, help="Nazwa endpointu")
    
    args = parser.parse_args()
    
    compile_pipeline(
        output_path=args.output_path,
        image_uri=args.image_uri,
        project=args.project,
        region=args.region,
        gcs_bucket=args.gcs_bucket,
        model_display_name=args.model_display_name,
        endpoint_name=args.endpoint_name
    )