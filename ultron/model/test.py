"""
{%- if messages[0]['role'] == 'system' -%}{%- set system_message = messages[0]['content'] -%}{%- set messages = messages[1:] -%}{%- else -%}{% set system_message = '' -%}{%- endif -%}{{ bos_token + system_message }}{%- for message in messages -%}{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{%- endif -%}{%- if message['role'] == 'user' -%}{{ ' USER: ' + message['content'] + '\n' }}{%- elif message['role'] == 'assistant' -%}{{ ' ASSISTANT: ' + message['content'] + eos_token + '\n' }}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{ ' ASSISTANT:' }}{% endif %}
"""
""" 
{{ bos_token }}
{% for message in messages %}
    {% if message['role'] != 'system' %}
        {{ " "+message['role'].upper() + ': '}}
    {% endif %}
    {# Render all images first #}
    {% for content in message['content'] | selectattr('type', 'equalto', 'image') %}
        {{ '<image>\n' }}
    {% endfor %}
    {# Render all text next #}
    {% if message['role'] != 'assistant' %}
        {% for content in message['content'] | selectattr('type', 'equalto', 'text') %}
            {{ content['text'] + ' '}}
        {% endfor %}
    {% else %}
        {% for content in message['content'] | selectattr('type', 'equalto', 'text') %}
            {% generation %}
            {{ content['text'] + eos_token + ' '  }}
            {% endgeneration %}
        {% endfor %}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ 'ASSISTANT:' }}
{% endif %}
"""

"{{ bos_token }}{% for message in messages %}{% if message['role'] != 'system' %}{{ " "+message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}{{eos_token}}"
