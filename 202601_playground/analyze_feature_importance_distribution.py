"""
Feature Importance 분포 분석 스크립트
- 특성 중요도가 한쪽에 몰려있는 현상 분석
- 개선 방안 제시
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

def load_feature_importance():
    """Feature importance 파일들을 로드"""
    base_dir = Path('feature_importance_results')
    
    # 각 모델별 중요도 로드
    catboost_df = pd.read_csv(base_dir / 'catboost_feature_importance.csv')
    lightgbm_df = pd.read_csv(base_dir / 'lightgbm_feature_importance.csv')
    xgboost_df = pd.read_csv(base_dir / 'xgboost_feature_importance.csv')
    common_df = pd.read_csv(base_dir / 'common_important_features.csv')
    
    return {
        'catboost': catboost_df,
        'lightgbm': lightgbm_df,
        'xgboost': xgboost_df,
        'common': common_df
    }

def analyze_concentration(df, model_name):
    """특성 중요도 집중도 분석"""
    importance = df['Importance'].values if 'Importance' in df.columns else df['Average_Importance'].values
    
    # 통계량 계산
    total_importance = importance.sum()
    top1_importance = importance[0]
    top3_importance = importance[:3].sum()
    top5_importance = importance[:5].sum()
    top10_importance = importance[:10].sum()
    
    # 집중도 비율
    top1_ratio = top1_importance / total_importance * 100
    top3_ratio = top3_importance / total_importance * 100
    top5_ratio = top5_importance / total_importance * 100
    top10_ratio = top10_importance / total_importance * 100
    
    # Gini 계수 (불평등도 측정)
    sorted_importance = np.sort(importance)[::-1]
    n = len(sorted_importance)
    cumsum = np.cumsum(sorted_importance)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_importance)) / (n * cumsum[-1]) - (n + 1) / n
    
    # 엔트로피 (다양성 측정, 높을수록 분산됨)
    normalized_importance = importance / total_importance
    normalized_importance = normalized_importance[normalized_importance > 0]  # 0 제거
    entropy = -np.sum(normalized_importance * np.log2(normalized_importance))
    max_entropy = np.log2(len(importance))  # 최대 엔트로피
    normalized_entropy = entropy / max_entropy  # 정규화된 엔트로피 (0~1)
    
    return {
        'model': model_name,
        'total_features': len(df),
        'total_importance': total_importance,
        'top1_importance': top1_importance,
        'top1_ratio': top1_ratio,
        'top3_ratio': top3_ratio,
        'top5_ratio': top5_ratio,
        'top10_ratio': top10_ratio,
        'gini_coefficient': gini,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'top_features': df.head(10)['Feature'].tolist()
    }

def categorize_features(feature_name):
    """특성을 카테고리별로 분류"""
    if '_x_' in feature_name:
        return '상호작용 (Interaction)'
    elif '_div_' in feature_name or '_ratio' in feature_name or 'ratio' in feature_name:
        return '비율 (Ratio)'
    elif '_freq' in feature_name:
        return '빈도 인코딩 (Frequency)'
    elif feature_name in ['study_hours', 'class_attendance', 'sleep_hours', 'age']:
        return '원본 수치형 (Original Numeric)'
    elif any(col in feature_name for col in ['sleep_quality', 'study_method', 'facility_rating', 
                                              'course', 'gender', 'internet_access', 'exam_difficulty']):
        if '_encoded' in feature_name:
            return 'Ordinal 인코딩'
        else:
            return 'One-Hot 인코딩'
    else:
        return '기타 (Other)'

def analyze_feature_categories(df):
    """특성 카테고리별 중요도 분석"""
    df = df.copy()
    df['Category'] = df['Feature'].apply(categorize_features)
    
    importance_col = 'Importance' if 'Importance' in df.columns else 'Average_Importance'
    
    category_stats = df.groupby('Category').agg({
        importance_col: ['sum', 'mean', 'count']
    }).round(4)
    
    category_stats.columns = ['Total_Importance', 'Mean_Importance', 'Count']
    category_stats = category_stats.sort_values('Total_Importance', ascending=False)
    
    return category_stats, df

def generate_recommendations(concentration_stats, category_stats):
    """개선 방안 제시"""
    recommendations = []
    
    # 집중도 분석
    avg_top1_ratio = np.mean([s['top1_ratio'] for s in concentration_stats.values()])
    avg_top3_ratio = np.mean([s['top3_ratio'] for s in concentration_stats.values()])
    avg_entropy = np.mean([s['normalized_entropy'] for s in concentration_stats.values()])
    
    recommendations.append("=" * 80)
    recommendations.append("📊 특성 중요도 집중도 분석 결과")
    recommendations.append("=" * 80)
    recommendations.append(f"\n1. 상위 1개 특성 집중도: 평균 {avg_top1_ratio:.1f}%")
    recommendations.append(f"2. 상위 3개 특성 집중도: 평균 {avg_top3_ratio:.1f}%")
    recommendations.append(f"3. 정규화된 엔트로피: {avg_entropy:.3f} (1.0에 가까울수록 분산됨)")
    
    # 해석
    recommendations.append("\n" + "-" * 80)
    recommendations.append("🔍 분석 해석")
    recommendations.append("-" * 80)
    
    if avg_top1_ratio > 40:
        recommendations.append("⚠️ 경고: 상위 1개 특성이 전체의 40% 이상을 차지합니다.")
        recommendations.append("   → 모델이 단일 특성에 과도하게 의존하고 있을 수 있습니다.")
    elif avg_top1_ratio > 30:
        recommendations.append("💡 주의: 상위 1개 특성이 30-40%를 차지합니다.")
        recommendations.append("   → 특성 엔지니어링을 통해 다른 특성의 중요도를 높일 수 있습니다.")
    else:
        recommendations.append("✅ 양호: 특성 중요도가 비교적 균등하게 분포되어 있습니다.")
    
    if avg_entropy < 0.7:
        recommendations.append(f"\n⚠️ 엔트로피가 낮음 ({avg_entropy:.3f}): 특성 중요도가 한쪽에 집중되어 있습니다.")
        recommendations.append("   → 더 많은 유의미한 특성을 생성할 필요가 있습니다.")
    else:
        recommendations.append(f"\n✅ 엔트로피가 양호함 ({avg_entropy:.3f}): 특성 중요도가 비교적 균등하게 분포되어 있습니다.")
    
    # 카테고리별 분석
    recommendations.append("\n" + "-" * 80)
    recommendations.append("📈 카테고리별 중요도 분석")
    recommendations.append("-" * 80)
    recommendations.append(category_stats.to_string())
    
    # 개선 방안
    recommendations.append("\n" + "=" * 80)
    recommendations.append("💡 특성 엔지니어링 개선 방안")
    recommendations.append("=" * 80)
    
    # 원본 수치형이 너무 강한 경우
    if '원본 수치형 (Original Numeric)' in category_stats.index:
        numeric_ratio = category_stats.loc['원본 수치형 (Original Numeric)', 'Total_Importance']
        if numeric_ratio > 0.5:
            recommendations.append("\n1. 원본 수치형 특성 변환 시도:")
            recommendations.append("   - 로그 변환: np.log1p(study_hours)")
            recommendations.append("   - 제곱근 변환: np.sqrt(study_hours)")
            recommendations.append("   - 구간화(Binning): study_hours를 5-10개 구간으로 분할")
            recommendations.append("   - 이상치 처리: IQR 기반 이상치 제거 또는 클리핑")
    
    # 상호작용 특성 확대
    if '상호작용 (Interaction)' in category_stats.index:
        interaction_ratio = category_stats.loc['상호작용 (Interaction)', 'Total_Importance']
        if interaction_ratio < 0.2:
            recommendations.append("\n2. 상호작용 특성 확대:")
            recommendations.append("   - 현재: study_hours × sleep_hours만 사용")
            recommendations.append("   - 추가 가능:")
            recommendations.append("     * study_hours × class_attendance")
            recommendations.append("     * age × study_hours")
            recommendations.append("     * (study_hours / sleep_hours) × class_attendance")
            recommendations.append("   - 다항식 특성: PolynomialFeatures(degree=2) 사용")
    
    # 비율 특성 확대
    if '비율 (Ratio)' in category_stats.index:
        ratio_ratio = category_stats.loc['비율 (Ratio)', 'Total_Importance']
        if ratio_ratio < 0.1:
            recommendations.append("\n3. 비율 특성 확대:")
            recommendations.append("   - 학습 효율: study_hours / sleep_hours")
            recommendations.append("   - 출석 효율: class_attendance / study_hours")
            recommendations.append("   - 시간 할당: study_hours / (study_hours + sleep_hours)")
            recommendations.append("   - 나이 대비 학습량: study_hours / age")
    
    # 범주형 특성 활용 개선
    onehot_importance = category_stats[category_stats.index.str.contains('One-Hot', na=False)]['Total_Importance'].sum()
    if onehot_importance < 0.15:
        recommendations.append("\n4. 범주형 특성 활용 개선:")
        recommendations.append("   - Target Encoding: 타겟 변수 기반 인코딩")
        recommendations.append("   - 범주형 조합: study_method × facility_rating")
        recommendations.append("   - 순서형 특성: 범주형 특성에 순서 부여")
    
    # 통계적 특성
    recommendations.append("\n5. 통계적 특성 추가:")
    recommendations.append("   - 그룹별 통계: study_method별 study_hours의 평균/표준편차")
    recommendations.append("   - 누적 특성: 과거 데이터 기반 누적 평균")
    recommendations.append("   - 순위 특성: 각 특성의 순위 또는 백분위수")
    
    # 시계열/순서 특성 (해당되는 경우)
    recommendations.append("\n6. 순서/시간 특성:")
    recommendations.append("   - 학습량 변화율: (현재 study_hours - 이전 study_hours)")
    recommendations.append("   - 누적 학습량: study_hours의 누적합")
    
    recommendations.append("\n" + "=" * 80)
    recommendations.append("⚠️ 주의사항")
    recommendations.append("=" * 80)
    recommendations.append("1. 특성 중요도가 한쪽에 몰려있다고 반드시 문제는 아닙니다.")
    recommendations.append("   - study_hours가 실제로 가장 예측력이 높은 특성일 수 있습니다.")
    recommendations.append("   - 시험 점수 예측에서 공부 시간이 가장 중요한 것은 자연스러울 수 있습니다.")
    recommendations.append("\n2. 과도한 특성 엔지니어링은 오히려 과적합을 유발할 수 있습니다.")
    recommendations.append("   - Cross-validation을 통해 검증하세요.")
    recommendations.append("   - Feature importance가 낮다고 무조건 제거하지 마세요.")
    recommendations.append("\n3. 모델 성능이 좋다면 특성 중요도 분포는 부차적입니다.")
    recommendations.append("   - 최종 목표는 예측 정확도입니다.")
    recommendations.append("   - 특성 중요도는 '설명력'을 위한 참고 자료입니다.")
    
    return "\n".join(recommendations)

def visualize_concentration(importance_dict, save_path='feature_importance_results/concentration_analysis.png'):
    """집중도 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('특성 중요도 집중도 분석', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. 각 모델별 상위 특성 중요도 (Bar Chart)
    ax1 = axes[0, 0]
    models = ['catboost', 'lightgbm', 'xgboost']
    top1_values = []
    top3_values = []
    top5_values = []
    
    for model in models:
        df = importance_dict[model]
        importance = df['Importance'].values if 'Importance' in df.columns else df['Average_Importance'].values
        total = importance.sum()
        top1_values.append(importance[0] / total * 100)
        top3_values.append(importance[:3].sum() / total * 100)
        top5_values.append(importance[:5].sum() / total * 100)
    
    x = np.arange(len(models))
    width = 0.25
    ax1.bar(x - width, top1_values, width, label='Top 1', color='#FF6B6B', alpha=0.8)
    ax1.bar(x, top3_values, width, label='Top 3', color='#4ECDC4', alpha=0.8)
    ax1.bar(x + width, top5_values, width, label='Top 5', color='#45B7D1', alpha=0.8)
    
    ax1.set_xlabel('모델', fontsize=12)
    ax1.set_ylabel('중요도 비율 (%)', fontsize=12)
    ax1.set_title('상위 특성의 중요도 집중도', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. 공통 중요도 분포 (Pareto Chart)
    ax2 = axes[0, 1]
    common_df = importance_dict['common'].head(15)
    importance_col = 'Average_Importance'
    y_pos = np.arange(len(common_df))
    
    bars = ax2.barh(y_pos, common_df[importance_col].values, color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(common_df['Feature'].values, fontsize=9)
    ax2.set_xlabel('평균 중요도', fontsize=12)
    ax2.set_title('공통 상위 15개 특성', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 누적 비율 라인 추가
    ax2_twin = ax2.twinx()
    cumulative = np.cumsum(common_df[importance_col].values) / common_df[importance_col].sum() * 100
    ax2_twin.plot(cumulative, y_pos, color='red', marker='o', linewidth=2, markersize=4)
    ax2_twin.set_ylabel('누적 비율 (%)', fontsize=12, color='red')
    ax2_twin.set_ylim(ax2.get_ylim())
    ax2_twin.grid(alpha=0.3, linestyle='--', axis='y')
    
    # 3. 카테고리별 중요도 (Pie Chart)
    ax3 = axes[1, 0]
    common_df_with_cat = importance_dict['common'].copy()
    common_df_with_cat['Category'] = common_df_with_cat['Feature'].apply(categorize_features)
    category_importance = common_df_with_cat.groupby('Category')[importance_col].sum().sort_values(ascending=False)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_importance)))
    wedges, texts, autotexts = ax3.pie(category_importance.values, 
                                        labels=category_importance.index,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        colors=colors)
    ax3.set_title('카테고리별 중요도 분포', fontsize=13, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    # 4. 엔트로피 비교
    ax4 = axes[1, 1]
    models = ['catboost', 'lightgbm', 'xgboost']
    entropies = []
    
    for model in models:
        df = importance_dict[model]
        importance = df['Importance'].values if 'Importance' in df.columns else df['Average_Importance'].values
        total = importance.sum()
        normalized = importance / total
        normalized = normalized[normalized > 0]
        entropy = -np.sum(normalized * np.log2(normalized))
        max_entropy = np.log2(len(importance))
        normalized_entropy = entropy / max_entropy
        entropies.append(normalized_entropy)
    
    bars = ax4.bar(models, entropies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    ax4.set_ylabel('정규화된 엔트로피', fontsize=12)
    ax4.set_title('모델별 특성 중요도 다양성 (엔트로피)', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='권장 최소값 (0.7)')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 값 표시
    for bar, entropy in zip(bars, entropies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{entropy:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 저장 완료: {save_path}")
    plt.close()

def main():
    print("=" * 80)
    print("📊 Feature Importance 집중도 분석 시작")
    print("=" * 80)
    
    # 데이터 로드
    importance_dict = load_feature_importance()
    
    # 집중도 분석
    concentration_stats = {}
    for model_name in ['catboost', 'lightgbm', 'xgboost']:
        stats = analyze_concentration(importance_dict[model_name], model_name)
        concentration_stats[model_name] = stats
        
        print(f"\n{model_name.upper()} 집중도 분석:")
        print(f"  상위 1개 특성 비율: {stats['top1_ratio']:.1f}%")
        print(f"  상위 3개 특성 비율: {stats['top3_ratio']:.1f}%")
        print(f"  정규화된 엔트로피: {stats['normalized_entropy']:.3f}")
        print(f"  최상위 특성: {stats['top_features'][0]}")
    
    # 카테고리별 분석
    category_stats, common_df_with_cat = analyze_feature_categories(importance_dict['common'])
    
    # 개선 방안 생성
    recommendations = generate_recommendations(concentration_stats, category_stats)
    
    # 결과 저장
    output_dir = Path('feature_importance_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'concentration_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(recommendations)
    
    print("\n" + recommendations)
    
    # 시각화
    visualize_concentration(importance_dict, output_dir / 'concentration_analysis.png')
    
    # 상세 통계 저장
    stats_df = pd.DataFrame(concentration_stats).T
    stats_df.to_csv(output_dir / 'concentration_statistics.csv', index=True, encoding='utf-8-sig')
    
    category_stats.to_csv(output_dir / 'category_statistics.csv', index=True, encoding='utf-8-sig')
    
    print(f"\n✅ 분석 완료!")
    print(f"   - 리포트: {output_dir / 'concentration_analysis_report.txt'}")
    print(f"   - 시각화: {output_dir / 'concentration_analysis.png'}")
    print(f"   - 통계: {output_dir / 'concentration_statistics.csv'}")

if __name__ == '__main__':
    main()
