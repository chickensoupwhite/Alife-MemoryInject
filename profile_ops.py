"""Persistent portrait aliases and hidden list entries; never edits source facts."""
import json
import os
import uuid


def load(index):
    path = os.path.join(os.path.dirname(os.path.abspath(index)), 'profile_overrides.json')
    if not os.path.isfile(path):
        return {'version': 1, 'aliases': {}, 'hidden': []}
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('画像设置格式不正确')
    if not isinstance(data.get('aliases'), dict) or not isinstance(data.get('hidden'), list):
        raise ValueError('画像别名或移除记录格式不正确')
    for source, target in data['aliases'].items():
        if not isinstance(source, str) or not isinstance(target, str) or not source.strip() or not target.strip():
            raise ValueError('画像别名不能为空')
        canonical(data, source)
    for item in data['hidden']:
        if not isinstance(item, dict) or item.get('kind') not in ('person', 'event', 'object') or not isinstance(item.get('name'), str):
            raise ValueError('画像移除记录格式不正确')
    return data


def canonical(settings, name):
    seen = set()
    while name in settings.get('aliases', {}):
        if name in seen:
            raise ValueError('画像别名存在循环')
        seen.add(name)
        name = settings['aliases'][name]
    return name


def family(settings, name):
    name = canonical(settings, name)
    return sorted({name} | {alias for alias in settings.get('aliases', {}) if canonical(settings, alias) == name})


def hidden(settings, kind, name):
    if kind == 'person':
        name = canonical(settings, name)
    return any(item['kind'] == kind and
               (canonical(settings, item['name']) if kind == 'person' else item['name']) == name
               for item in settings.get('hidden', []))


def apply(req):
    # Reuse the existing management boundary: writes stay in this role's MemoryFacts.
    import audit_ops
    import readonly_query
    from textsim import classify_subject
    try:
        index, factsdir = audit_ops._audit_plugin_paths(req)
        settings = load(index)
        action = str(req.get('action') or '')
        kind = str(req.get('kind') or 'person')
        name = str(req.get('name') or '').strip()
        if kind not in ('person', 'event', 'object') or not name or len(name) > 64:
            raise ValueError('请选择有效的画像名称和类别')
        if action in ('merge', 'rename'):
            if kind != 'person':
                raise ValueError('称呼合并仅适用于人物画像')
            target = str(req.get('target') or '').strip()
            if not target or len(target) > 64:
                raise ValueError('请输入 1–64 字的目标名称')
            source = canonical(settings, name)
            target = canonical(settings, target)
            source_profile = readonly_query.db_profile_readonly(dict(index=index, name=source, kind='person', limit=1))
            if not source_profile or not source_profile.get('found'):
                raise ValueError('原人物画像已不存在或已被移除，请刷新后重试')
            if action == 'merge':
                destination = readonly_query.db_profile_readonly(dict(index=index, name=target, kind='person', limit=1))
                if not destination or not destination.get('found'):
                    raise ValueError('合并目标必须是现有的人物画像')
            elif classify_subject(target)['kind'] not in ('person', 'unknown') or any(c in target for c in '/\\:|、'):
                raise ValueError('新称呼不能是软件、事件、代词、组合主体或路径')
            elif target != source:
                destination = readonly_query.db_profile_readonly(dict(index=index, name=target, kind='person', limit=1))
                if destination and destination.get('found'):
                    raise ValueError('该称呼已有独立画像，请使用“合并”并核对目标人物')
            if source != target:
                for alias in family(settings, source):
                    if alias != target:
                        settings['aliases'][alias] = target
            name = target
        elif action == 'hide':
            name = canonical(settings, name) if kind == 'person' else name
            if not hidden(settings, kind, name):
                settings['hidden'].append({'kind': kind, 'name': name})
        elif action == 'restore':
            name = canonical(settings, name) if kind == 'person' else name
            settings['hidden'] = [item for item in settings['hidden']
                                  if not (item['kind'] == kind and
                                  (canonical(settings, item['name']) if kind == 'person' else item['name']) == name)]
        elif action == 'unmerge':
            if kind != 'person':
                raise ValueError('只有人物称呼可以拆分')
            settings['aliases'].pop(name, None)
        else:
            raise ValueError('不支持的画像操作')
        path = os.path.join(factsdir, 'profile_overrides.json')
        temporary = path + '.' + uuid.uuid4().hex + '.tmp'
        try:
            with open(temporary, 'w', encoding='utf-8', newline='\n') as stream:
                json.dump(settings, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return {'ok': True, 'name': name, 'kind': kind, 'action': action, 'management': settings}
    except (OSError, ValueError) as exc:
        return {'ok': False, 'error': str(exc)}
