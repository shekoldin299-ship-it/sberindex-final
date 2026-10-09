"""Official-key geography and transparently selected local event illustrations."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import load_data
from experiment import ROOT,write_json
from detection import innovations
from upgrade_v2 import robust_chart
from upgrade_v3 import factor_scores,panel_alarms

def main(dictionary):
    out=ROOT/'v3_results';out.mkdir(exist_ok=True)
    if dictionary:
        raw=pd.read_excel(dictionary)
        columns=['territory_id','municipal_district_name_short','municipal_district_name','region_name','oktmo','year_from','year_to']
        raw[columns].to_csv(ROOT/'data/municipality_dictionary.csv',index=False)
        write_json(ROOT/'data/municipality_dictionary_source.json',{'source':'СберИндекс','url':'https://s.sber.ru/GthXk7',
            'dataset_page':'https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities',
            'retrieved':'2026-10-08','original_filename':'t_dict_municipal_districts.xlsx',
            'xlsx_sha256':hashlib.sha256(Path(dictionary).read_bytes()).hexdigest(),
            'join':'Official territory_id; latest record beginning no later than 2024. Historical boundaries may differ; no matching by guessed names.',
            'verification_limit':'Public spending endpoint returned HTTP 502; numerical series identity not independently cross-checked.'})
    raw=pd.read_csv(ROOT/'data/municipality_dictionary.csv')
    _,panel,dates,_=load_data(ROOT/'data/consumption.parquet')
    names=raw[raw.year_from.le(2024)].sort_values(['territory_id','year_from','year_to']).drop_duplicates('territory_id',keep='last')
    ids=pd.DataFrame({'territory_id':panel.index.get_level_values(0).unique()})
    names=ids.merge(names,on='territory_id',how='left',validate='one_to_one')
    names['name_status']=np.where(names.municipal_district_name.isna(),'unmapped',np.where(names.year_to.lt(2024),'historical_record','official_key_2024'))
    names.to_csv(out/'municipality_names.csv',index=False)
    # Cities selected by independently documented April flood, not detector success.
    events=[{'territory_id':1673,'name':'Орск','published_at':'2024-04-06','available_at':'2024-04-07','event_month':'2024-04',
             'summary':'МЧС сообщает об эвакуации после размыва дамбы.','url':'https://56.mchs.gov.ru/deyatelnost/press-centr/novosti/5249015'},
            {'territory_id':1665,'name':'Оренбург','published_at':'2024-04-17','available_at':'2024-04-18','event_month':'2024-04',
             'summary':'МЧС сообщает о снижении уровня воды после прохождения паводковой волны.','url':'https://mchs.gov.ru/deyatelnost/press-centr/novosti/5257170'},
            {'territory_id':1333,'name':'Курган','published_at':'2024-04-17','available_at':'2024-04-18','event_month':'2024-04',
             'summary':'МЧС сообщает о подходе паводковой волны к областному центру.','url':'https://mchs.gov.ru/deyatelnost/press-centr/novosti/5257170'}]
    write_json(ROOT/'data/local_events.json',events)
    allcat=panel.xs('Все категории',level=1)
    # Prefix-time statistic: each month uses only that month and the preceding year.
    growth=np.log1p(allcat.iloc[:,12:].to_numpy())-np.log1p(allcat.iloc[:,:12].to_numpy())
    factor=np.nanmedian(growth,axis=0);spread=np.maximum(1.4826*np.nanmedian(abs(growth-factor),axis=0),.02)
    cases=[];rows=[]
    for event in events:
        name=names[names.territory_id.eq(event['territory_id'])].iloc[0]
        assert name.municipal_district_name_short==event['name']
        y=allcat.loc[event['territory_id']].to_numpy();z=innovations(y)
        alarms={'cusum_v1':[r['t'] for r in robust_chart(z,'cusum_v1',8) if r['alarm']],
                'v2_multiscale':[r['t'] for r in robust_chart(z,'multiscale',5) if r['alarm']],
                'panel3_experimental':panel_alarms(factor_scores(y,factor,spread),3,4)}
        relative=np.r_[np.full(12,np.nan),(np.log1p(y[12:])-np.log1p(y[:-12])-factor)*100]
        cases.append({**event,'region':name.region_name,'alarms':{m:[str(dates[t].date()) for t in ts] for m,ts in alarms.items()},
                      'relative_growth_change_apr_jun_vs_jan_mar':float(np.nanmean(relative[15:18])-np.nanmean(relative[12:15]))})
        for t,d in enumerate(dates):
            rows.append({'territory_id':event['territory_id'],'name':event['name'],'month':str(d.date()),'actual':y[t],
                         'relative_log_yoy_pp':relative[t],**{m:t in ts for m,ts in alarms.items()},
                         'news_known_after_month':pd.Timestamp(event['available_at']) < d+pd.offsets.MonthBegin()})
    pd.DataFrame(rows).to_csv(out/'local_case_history.csv',index=False)
    write_json(out/'local_cases.json',cases)
    lines=['# Локальные кейсы: паводок апреля 2024 года','',
           'Три города выбраны по опубликованным сообщениям МЧС, до оценки успешности сигналов. Это иллюстрации, не независимая разметка всех экономических шоков. Новостные записи не используются для настройки прогноза или детектора.',
           '',f'Названия присоединены по официальному territory_id: {names.municipal_district_name.notna().sum()} из {len(names)} ID. Для изменявшихся МО сохранён статус исторической записи. Численное совпадение с публичной выгрузкой расходов отдельно не проверено: её API вернул HTTP 502.','',
           'Сигнал датирован месяцем наблюдения. Он мог быть получен только после публикации расходов за этот месяц; точный лаг публикации неизвестен. Новость считается доступной со следующего дня после публикации. Совпадение новости и сигнала не доказывает причинность.','']
    for c in cases:
        lines += [f"## {c['name']} — {c['region']}",'',f"ID {c['territory_id']}. {c['summary']} [МЧС, {c['published_at']}]({c['url']}).",'',
                  f"Изменение среднего относительного годового лог-прироста за апрель–июнь против января–марта: {c['relative_growth_change_apr_jun_vs_jan_mar']:.2f} п.п. Это описательная разность, а не причинный эффект паводка.",'']
        for m,ds in c['alarms'].items(): lines.append(f"- {m}: {', '.join(ds) if ds else 'сигналов нет'}.")
        lines+=['','Снижение расходов может сочетаться с эвакуацией, изменением покупок и другими факторами; выплаты и восстановительные покупки могут действовать в обратную сторону. Без контрольного исследования причину не устанавливаем.','']
    (ROOT/'LOCAL_CASES.md').write_text('\n'.join(lines))
    print('Geography and local cases saved',len(names),len(cases))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dictionary');a=p.parse_args();main(a.dictionary)
