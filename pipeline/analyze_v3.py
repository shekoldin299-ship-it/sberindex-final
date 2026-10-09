"""Independent checks and Russian report assembled from saved measurements."""
import hashlib,json
import numpy as np
import pandas as pd
from experiment import ROOT,write_json
from benchmark import load_data

OUT=ROOT/'v3_results'
def table(df):
    def cell(x):return f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    w=pd.read_parquet(OUT/'predictions.parquet')
    _,panel,_,_=load_data(ROOT/'data/consumption.parquet')
    keys=['territory_id','category','origin','horizon']
    assert not w.duplicated(keys).any()
    actual=np.array([panel.loc[(r.territory_id,r.category)].iloc[r.origin+r.horizon-1] for r in w.itertuples()])
    np.testing.assert_allclose(actual,w.actual)
    assert np.isfinite(w[['v2','v3','lower','upper']]).all().all()
    assert (w.lower<=w.v3).all() and (w.upper>=w.v3).all()
    for name,digest in json.loads((OUT/'signature.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    metrics=pd.read_csv(OUT/'metrics.csv')
    for r in metrics.itertuples():
        g=w[w.horizon.eq(r.horizon)];assert abs(abs(g[r.model]-g.actual).mean()-r.mae)<1e-7
    tuning=pd.read_parquet(OUT/'tuning_candidates.parquet');policy=json.loads((OUT/'policy.json').read_text())
    for key,p in policy.items():
        h,cat=key.split('|');g=tuning[tuning.horizon.eq(int(h))&tuning.category.eq(cat)]
        for name,value in p['tuning_mae'].items():assert abs(abs(g[name]-g.actual).mean()-value)<1e-7
        chosen=min(p['tuning_mae'],key=p['tuning_mae'].get)
        if int(h)==12 or p['tuning_mae'][chosen]>.98*p['tuning_mae']['v2']:chosen='v2'
        assert chosen==p['model']
    summary=[];intervals=[]
    for cohort,df in [('all',w),('remaining',w[w.cohort.eq('remaining')]),('v2_test',w[w.cohort.eq('v2_test')])]:
        for h,g in df.groupby('horizon'):
            v2=abs(g.v2-g.actual).mean();v3=abs(g.v3-g.actual).mean()
            summary.append({'cohort':cohort,'h':int(h),'n':len(g),'V2_MAE':v2,'V3_MAE':v3,'change_pct':100*(v3/v2-1)})
            intervals.append({'cohort':cohort,'h':int(h),'coverage':float(g.actual.between(g.lower,g.upper).mean()),'width':float((g.upper-g.lower).mean())})
    s=pd.DataFrame(summary);s.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'interval_metrics.csv',index=False)
    paired=pd.read_parquet(OUT/'paired_prophet.parquet')
    paired_metrics=paired.groupby('horizon').apply(lambda g:pd.Series({n:abs(g[n]-g.actual).mean() for n in ['v2','v3','prophet_default']}),include_groups=False).reset_index()
    paired_metrics.to_csv(OUT/'paired_prophet_metrics.csv',index=False)
    det=pd.read_csv(OUT/'detector_metrics.csv');dp=json.loads((OUT/'detector_policy.json').read_text())
    assert not set(dp['calibration_ids'])&set(dp['test_ids'])
    for item in json.loads((OUT/'detector_cases.json').read_text()):
        tp=fp=fn=0
        for c in item['cases']:
            hit=c['change'] is not None and any(c['change']<=t<=c['change']+2 for t in c['alarms'])
            tp+=hit;fp+=len(c['alarms'])-hit;fn+=c['change'] is not None and not hit
        row=det[det.split.eq('test')&det.method.eq(item['method'])].iloc[0]
        assert (tp,fp,fn)==(row.tp,row.fp,row.fn)
    missing=[{'origin':c,'series_without_history':int((~np.isfinite(panel.iloc[:,:c]).any(axis=1)).sum())} for c in (12,18,21,23)]
    write_json(OUT/'integrity.json',{'status':'PASS','pairs':len(w),'territories':int(w.territory_id.nunique()),'missing_history':missing,'checks':['source signature','unique keys','actual targets','MAE recalculation','tuning-only policy','finite forecasts','interval ordering','detector disjoint groups and TP/FP/FN']})
    allcat=pd.read_csv(OUT/'category_metrics.csv');allcat=allcat[allcat.category.eq('Все категории')&allcat.model.isin(['v2','v3','seasonal_scaled'])][['model','horizon','n','mae']]
    report=f'''# V3: расширенная проверка и собственная модель общей динамики

8 октября 2026. Реализация написана самостоятельно. V1/V2 сохранены.

## Что изменено
Общая компонента — медиана логарифмов расходов внутри категории по наблюдаемой панели. Из неё выделяются сезонность и годовой рост; местный уровень оценивается относительно общего фактора. Это статистический ориентир по муниципалитетам, не официальный национальный индекс. Новые смеси с V2 выбираются по прежним 50 tuning-МО; минимум улучшения 2%. h12 остаётся V2. Методика и ограничения: V3_PROTOCOL.md.

## Сопоставимое расширенное сравнение
Отрицательное change_pct означает меньшую ошибку V3. `remaining` исключает 50 tuning, 100 calibration, 100 V2-test и 10 pilot МО. Все группы используют прежние календарные периоды: независимый временной тест отсутствует.

{table(s)}

## Отдельно «Все категории», вся доступная панель
Эти рублёвые ошибки нельзя смешивать со средним по шести категориям или с результатами на других датах.

{table(allcat)}

## Prophet: только общие ключи сохранённой выборки V2
Prophet не пересчитывался по всей панели. Нельзя делить его выборочную MAE на полнопанельную MAE нашей модели.

{table(paired_metrics)}

## Интервалы V3
Перекалиброваны на прежних 100 calibration-МО; номинал 90%. История уже изучена, гарантии покрытия на будущем не заявляются. Покрытие на calibration не служит тестом.

{table(pd.DataFrame(intervals))}

## Детектирование: реальные ряды с искусственно внесёнными событиями
300 тестовых МО, 7 сценариев на каждый; 1500 структурных изменений. Раздельные 300 calibration-МО. Окно обнаружения 0–2 месяца; повторные и ранние сигналы учитываются как FP. Порог нового семейства выбирается по calibration при контрольных тревогах не более 5%; фактическая частота на тесте различается. Это общий бюджет ложных тревог, не точное равенство частот.

{table(det[det.split.eq('test')].drop(columns='split'))}

`control_rate` — доля неизменённых контрольных рядов с хотя бы одной тревогой; эти реальные ряды могут содержать неизвестные события. `median_delay` относится только к обнаруженным изменениям. Результат не доказывает предсказание будущих кризисов. Прежние детекторы сохранены; новый метод не объявляется лучшим по всем метрикам.

## Полнота и воспроизводимость
Проверено {len(w):,} пар, {w.territory_id.nunique()} МО. Исключения без истории по origin: {missing}. Все кандидаты, политика, категории, группы и сигналы сохранены в v3_results. Проверка: python -m unittest test_v3; python analyze_v3.py. Полный расчёт: python upgrade_v3.py --phase forecast; python upgrade_v3.py --phase detector; python analyze_v3.py.

## Что пока не подтверждено
Названия МО присоединены по официальному territory_id и исторической версии справочника; численное сопоставление с публичной выгрузкой расходов не выполнено из-за HTTP 502 её API. Три локальных кейса МЧС и ограничения представлены в LOCAL_CASES.md. h12 оценивается только на одной дате. Новые кандидаты не улучшают автоматически каждую категорию и каждую территорию.
'''
    (ROOT/'V3_REPORT.md').write_text(report)
    print(s.to_string(index=False));print('Independent V3 validation PASS')

if __name__=='__main__':main()
