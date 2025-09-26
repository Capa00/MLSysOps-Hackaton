"""Plugin module for custom policies - notify function."""
from __future__ import print_function

import inspect
import re
import time
import logging
import random
import requests
from mlsysops.logger_util import logger
from mlsysops.utilities import evaluate_condition
import traceback
def initialize():
    print(f"Initializing policy {inspect.stack()[1].filename}")

    initialContext = {
        "telemetry": {
            "metrics": ["node_cpu_seconds_total", "frame_classify_latency"],
            "system_scrape_interval": "1s"
        },
        "mechanisms": ["fluidity"],
        "packages": ["requests"],
        "configuration": {
            "analyze_interval": "10s"
        },
        "scope": "application",
    }

    return initialContext

async def analyze(context, application_description, system_description, mechanisms, telemetry, ml_connector):

    # policy handles single policy, always an array with a single application
    application_spec = application_description[0]['spec']
    application_components = application_spec['components']
    logger.info("Running analyzing two... ")
    for application_component in application_components:
        if "qos_metrics" not in application_component:
            continue
        component_metrics = application_component['qos_metrics']
        for component_metric in component_metrics:
            metric_name = component_metric['application_metric_id']
            # Get latest values from telemetry data
            try:
                logger.info("Getting telemetry")
                latest_telemetry_df = await telemetry['query']()
                logger.info(latest_telemetry_df)
            except Exception as e:
                logger.error(traceback.format_exc())
                continue
            component_metric_target = component_metric['target']
            component_measured_metric = latest_telemetry_df[metric_name].values[0]
            logger.debug(
                f"metric {metric_name} Target {component_metric_target} measurement {component_measured_metric} ")

            if component_measured_metric is None:
                continue

            ml_deployment_id = "tadfasdfas"  # it works model version
            ml_connector_endpoint = "team-grup"

            resp = requests.get(f"{ml_connector_endpoint}/deployment/get/status/{ml_deployment_id}")
            # ML mode
            logger.info("trying ML inference")
            if resp.status == "ready":
                try:
                    inference_endpoint = resp.inference_endpoint
                    value = latest_telemetry_df['timestamp']
                    payload = {
                        "data": [
                            {"time": "2025-10-09 22:20:00", "mls-compute-vm3_cpu_avg": "0.00128",
                             "mls-compute-vm3_free_memory": "300.0"},
                        ],
                        "is_fun": true,
                        "explanation": false
                    }
                    # payload = {
                    #     "data": latest_telemetry_df[-10:].to_dict(orient='records')
                    #
                    #         # {"feat1_row1": [], "feat2_row1": []},
                    #         # {"feat1_row2":  [], "feat2_row2":  []},
                    #         , # timestamp in string format
                    #         # cpu utilization all cores values: 0 - 1.0
                    #         # available memory values: bytes
                    #     #[],  # input features -
                    #     "is_fun": False,
                    #     "explanation": False
                    # }
                    resp = requests.post(f"{inference_endpoint}/prediction", json=payload)
                    # expected format: {  "inference": "[34,35,36]" }
                    response_json = resp.json()
                    inference_result = response_json['inference']
                    logger.warning(response_json)
                    # auxiliary method - placeholder
                    return True

                except Exception as e:
                    logger.error(f"Error at ML inference {e}")
                    pass

            # Heuristic mode
            if evaluate_condition(component_metric_target,component_measured_metric, component_metric['relation']):
                # even one telemetry metric is not fulfilled, return true
                return True, context

    return True, context


async def plan(context, application_description, system_description, mechanisms, telemetry, ml_connector):
    # Do not touch this code
    application = application_description[0]
    # check if in the state the client app has been placed
    # use fluidity state for that
    components_state = mechanisms['fluidity']['state']['applications'][application_description[0]['name']]['components']
    context['name'] = application['name']
    context['spec'] = application['spec']

    plan_result = {}
    plan_result['name'] = context['name']
    plan_result['deployment_plan'] = {}

    for component in application['spec']['components']:
        comp_name = component['metadata']['name']
        node_placement = component.get("node_placement")

        if comp_name == "detector":
            continue

        current_node_placed = components_state[comp_name]['node_placed']

        if current_node_placed is not None:
            # component is placed, move it to another
            available_nodes = [node for node in system_description['MLSysOpsCluster']['nodes'] if node != current_node_placed]
            node_to_place = random.choice(available_nodes)

            new_component_plan = {
                "action": "move",
                "target_host": node_to_place,
                "src_host": current_node_placed,
            }
            if comp_name not in plan_result['deployment_plan']:
                plan_result['deployment_plan'][comp_name] = []

            plan_result['deployment_plan'][comp_name].append(new_component_plan)

    if len(plan_result['deployment_plan'].keys()) == 0:
        return {}, context # no plan produced

    plan_result['deployment_plan']['initial_plan'] = False

    new_plan = {
        "fluidity": plan_result,
    }

    logger.info('New plan %s', new_plan)

    return new_plan, context
