import argparse
import os
from google.cloud import aiplatform

def run_pipeline(
    pipeline_path: str,
    project_id: str,
    region: str,
    pipeline_root: str,
    pipeline_name: str,
    service_account: str
):

    aiplatform.init(project=project_id, location=region)
    
    job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path=pipeline_path,
        pipeline_root=pipeline_root,
        enable_caching=False,
        parameter_values={}, 
    )
    
    job.submit(service_account=service_account)
    
    print(f"Pipeline uruchomiony: {job.display_name}")
    print(f"ID uruchomienia: {job.name}")
    print(f"Link do konsoli: {job.gca_resource.name}")
    
    job.wait()
    
    tasks = job.gca_resource.job_detail.task_details
    last_task = tasks[-1]
    
    print("Pipeline zakończony!")
    print(f"Status: {job.state}")
    
    if job.state == "PIPELINE_STATE_SUCCEEDED":
        endpoint_url = last_task.outputs.artifacts.artifacts[-1].uri
        print(f"URL endpointu: {endpoint_url}")
        return endpoint_url
    else:
        print("Pipeline nie zakończył się sukcesem, sprawdź logi w konsoli Google Cloud.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchamia pipeline trenowania i wdrażania modelu w Vertex AI")
    parser.add_argument("--pipeline-path", required=True, help="Ścieżka do pliku pipeline JSON")
    parser.add_argument("--project-id", required=True, help="ID projektu Google Cloud")
    parser.add_argument("--region", required=True, help="Region Google Cloud")
    parser.add_argument("--pipeline-root", required=True, help="Katalog roboczy pipeline (GCS)")
    parser.add_argument("--pipeline-name", required=True, help="Nazwa uruchomienia pipeline")
    parser.add_argument("--service-account", required=True, help="Service account do uruchomienia pipeline")
    
    args = parser.parse_args()
    
    run_pipeline(
        pipeline_path=args.pipeline_path,
        project_id=args.project_id,
        region=args.region,
        pipeline_root=args.pipeline_root,
        pipeline_name=args.pipeline_name,
        service_account=args.service_account
    )