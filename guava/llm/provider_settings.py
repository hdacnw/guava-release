"""Shared hosted-provider defaults; credentials are read only from the environment."""
DEFAULTS = {
    'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY', 'openai/gpt-5.4'),
    'openai': ('https://api.openai.com/v1/responses', 'OPENAI_API_KEY', 'gpt-5.4'),
    'nvidia': ('https://inference-api.nvidia.com/v1/responses', 'NVIDIA_API_KEY', 'openai/openai/gpt-5.4'),
}


def add_provider_arguments(parser):
    parser.add_argument('--provider', choices=tuple(DEFAULTS), default='openrouter')
    parser.add_argument('--base-url')
    parser.add_argument('--model', help='Provider-specific vision model ID.')
    parser.add_argument('--api-key-env')


def resolve_provider_arguments(args):
    endpoint, key_env, model = DEFAULTS[args.provider]
    args.base_url = args.base_url or endpoint
    args.api_key_env = args.api_key_env or key_env
    args.model = args.model or model
