/**
 * Hackathon AI Judge - Cloudflare Agents with Orchestrator-Workers Pattern
 *
 * This worker implements an AI-powered hackathon judging system using the Orchestrator-Workers pattern:
 * - Orchestrator: Creates 3 evaluation tasks (business feasibility, business value, technical validity)
 * - Workers: Each worker uses a different AI model to evaluate one specific criterion
 * - Integration: Combines all evaluations into a standardized score format
 *
 * Usage:
 * POST request with JSON body: { "idea": "your idea", "technology": "tech stack", "theme": "optional theme" }
 * Returns: JSON formatted evaluation with total score and detailed breakdown
 *
 * - Run `npm run dev` in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run `npm run deploy` to publish your worker
 *
 * Learn more at https://developers.cloudflare.com/agents/
 */

import { Agent, AgentNamespace, getAgentByName } from 'agents';
import { streamText, generateObject } from 'ai';
import { createWorkersAI } from 'workers-ai-provider';
import { z } from 'zod';

// 型定義とスキーマ
interface Env {
	MyAgent: AgentNamespace<MyAgent>;
	AI: Ai;
	CORS_ALLOWED_ORIGINS?: string;
	IMAGE_API_URL?: string;
}

// AIEvaluationRequestインターフェース
export interface AIEvaluationRequest {
	theme: string;
	direction: string;
	idea: string;
	techNames: string[];
}

// AIEvaluationResponseインターフェース
export interface AIEvaluationResponse {
	totalScore: number;
	comment: string;
	generatedImageUrl?: string;
	breakdown: {
		criteria1: number;    // 採点項目1（20点満点）
		criteria2: number;    // 採点項目2（20点満点）
		criteria3: number;    // 採点項目3（20点満点）
	};
}

// 評価タスクのスキーマ
const EvaluationTaskSchema = z.object({
	taskId: z.string(),
	criterion: z.enum(['business_feasibility', 'business_value', 'technical_validity']),
	maxScore: z.number(),
	description: z.string(),
});

// 評価結果のスキーマ
const EvaluationResultSchema = z.object({
	taskId: z.string(),
	criterion: z.string(),
	score: z.number(),
	maxScore: z.number(),
	reason: z.string(),
	success: z.boolean(),
	error: z.string().optional(),
});

// 最終評価結果のスキーマ
const FinalEvaluationSchema = z.object({
	totalScore: z.number(),
	businessFeasibility: z.object({
		score: z.number(),
		reason: z.string(),
	}),
	businessValue: z.object({
		score: z.number(),
		reason: z.string(),
	}),
	technicalValidity: z.object({
		score: z.number(),
		reason: z.string(),
	}),
});


// 型定義
type EvaluationTask = z.infer<typeof EvaluationTaskSchema>;
type EvaluationResult = z.infer<typeof EvaluationResultSchema>;


/**
 * 評価オーケストレーター: 3つの評価観点のタスクを作成・配信
 */
class EvaluationOrchestrator {
	constructor(private workersai: ReturnType<typeof createWorkersAI>) { }

	createEvaluationTasks(idea: string, technology: string, theme?: string, direction?: string): EvaluationTask[] {
		const timestamp = Date.now();

		return [
			{
				taskId: `business-feasibility-${timestamp}`,
				criterion: 'business_feasibility' as const,
				maxScore: 20,
				description: 'ビジネス的実現性（20点）: ビジネスモデルとしての実現可能性',
			},
			{
				taskId: `business-value-${timestamp}`,
				criterion: 'business_value' as const,
				maxScore: 20,
				description: 'ビジネス的価値（20点）: 市場価値と収益性の見込み',
			},
			{
				taskId: `technical-validity-${timestamp}`,
				criterion: 'technical_validity' as const,
				maxScore: 20,
				description: '技術の妥当性（20点）: 技術選択と実装の適切さ',
			}
		];
	}
}

/**
 * 評価ワーカー: 各観点で独立した評価を実行
 */
class EvaluationWorker {
	constructor(private workersai: ReturnType<typeof createWorkersAI>) { }

	/**
	 * 評価観点に応じて適切なAIモデルを選択
	 */
	private selectModelForCriterion(criterion: string): ReturnType<typeof this.workersai> {
		// JSON MODEサポートモデルを使用
		switch (criterion) {
			case 'business_feasibility':
				// ビジネス的実現性: ビジネス分析に優れたモデル
				return this.workersai("@cf/meta/llama-3.1-8b-instruct");
			case 'business_value':
				// ビジネス的価値: 市場価値評価に優れたモデル
				return this.workersai("@cf/google/gemma-3-12b-it");
			case 'technical_validity':
				// 技術の妥当性: 技術評価に優れたモデル
				return this.workersai("@cf/qwen/qwen2.5-coder-32b-instruct");
			default:
				return this.workersai("@cf/meta/llama-3.1-8b-instruct");
		}
	}

	/**
	 * 評価観点別のシステムプロンプトを取得
	 */
	private getSystemPromptForCriterion(criterion: string): string {
		const basePrompt = 'あなたはビジネス・技術の専門審査員です。公正で客観的な評価を行ってください。レスポンスは必ず有効なJSON形式のみで返してください。 理由は一言で';

		switch (criterion) {
			case 'business_feasibility':
				return `${basePrompt} ビジネス的実現性を専門的に評価してください。収益モデル、市場参入の容易さ、競合優位性、運営コストなどを重視してください。`;
			case 'business_value':
				return `${basePrompt} ビジネス的価値を専門的に評価してください。市場規模、収益性、成長ポテンシャル、社会的価値などを重視してください。`;
			case 'technical_validity':
				return `${basePrompt} 技術の妥当性を専門的に評価してください。技術選択の適切さ、実装の現実性、拡張性、保守性などを重視してください。`;
			default:
				return basePrompt;
		}
	}

	async evaluateTask(task: EvaluationTask, idea: string, technology: string, theme?: string, direction?: string): Promise<EvaluationResult> {
		try {
			const model = this.selectModelForCriterion(task.criterion);
			const systemPrompt = this.getSystemPromptForCriterion(task.criterion);

			const themeContext = theme ? `テーマ: ${theme}\n` : '';
			const directionContext = direction ? `方向性: ${direction}\n` : '';

			const evaluationStream = await streamText({
				model: model,
				system: `${systemPrompt} JSON形式のみで回答。理由は一言で。`,
				prompt: `${themeContext}${directionContext}アイデア: ${idea}
技術: ${technology}

評価観点: ${task.description}
最大点: ${task.maxScore}点

JSON回答:
{"score": [0-${task.maxScore}の整数], "reason": "一言コメント"}`,
			});

			const evaluationText = await evaluationStream.text;
			console.log(`🔍 AI Response for ${task.criterion}:`, evaluationText.substring(0, 200) + '...');

			// 改善されたJSON解析処理
			let evaluation = null;

			try {
				// 方法1: 完全なJSONオブジェクトを解析
				const jsonMatch = evaluationText.match(/\{[^{}]*"score"\s*:\s*\d+[^{}]*"reason"\s*:[^{}]*\}/);
				if (jsonMatch) {
					evaluation = JSON.parse(jsonMatch[0]);
				}
			} catch (e) {
				// JSON解析失敗時は次の方法へ
			}

			// 方法2: scoreとreasonを個別に抽出（より安全）
			if (!evaluation) {
				const scoreMatch = evaluationText.match(/"score"\s*:\s*(\d+)/);
				let reasonMatch = evaluationText.match(/"reason"\s*:\s*"([^"]{1,20})"/); // 20文字制限

				// reasonが見つからない場合の追加パターン
				if (!reasonMatch) {
					reasonMatch = evaluationText.match(/理由[:\s]*([^\n]{1,15})/);
				}

				if (scoreMatch) {
					evaluation = {
						score: Math.max(0, Math.min(task.maxScore, parseInt(scoreMatch[1]))),
						reason: reasonMatch ? reasonMatch[1].trim() : '評価完了'
					};
				}
			}

			// 方法3: 数値のみ抽出（最小限のフォールバック）
			if (!evaluation) {
				const scorePattern = /(\d+)\s*[点\/]/;
				const scoreMatch = evaluationText.match(scorePattern);

				if (scoreMatch) {
					const score = Math.max(0, Math.min(task.maxScore, parseInt(scoreMatch[1])));
					evaluation = {
						score: score,
						reason: `${task.criterion}評価: ${score}点`
					};
				}
			}

			if (evaluation) {
				const score = Math.max(0, Math.min(task.maxScore, parseInt(evaluation.score) || 0));
				console.log(`✅ Successfully parsed ${task.criterion}: ${score}/${task.maxScore}点`);

				return {
					taskId: task.taskId,
					criterion: task.criterion,
					score: score,
					maxScore: task.maxScore,
					reason: evaluation.reason || '評価理由が提供されませんでした',
					success: true,
				};
			} else {
				// フォールバック: 中間点を返す
				console.warn(`⚠️ JSON parse failed for ${task.criterion}, using fallback. Raw response:`, evaluationText.substring(0, 300));
				return {
					taskId: task.taskId,
					criterion: task.criterion,
					score: Math.floor(task.maxScore / 2),
					maxScore: task.maxScore,
					reason: `${task.description}の評価中にレスポンス解析エラーが発生しましたが、中間的な評価を行いました。`,
					success: true,
				};
			}
		} catch (error) {
			return {
				taskId: task.taskId,
				criterion: task.criterion,
				score: 0,
				maxScore: task.maxScore,
				reason: '',
				success: false,
				error: error instanceof Error ? error.message : 'Unknown error',
			};
		}
	}
}

/**
 * 画像生成機能: 外部APIで画像を生成
 */
class ImageGenerator {
	private readonly imageApiUrl: string;

	constructor(imageApiUrl?: string) {
		this.imageApiUrl = imageApiUrl;
	}

	async generateImage(idea: string): Promise<string | null> {
		try {
			console.log('🎨 画像生成開始:', idea);
			const response = await fetch(this.imageApiUrl, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ idea }),
			});

			if (!response.ok) {
				console.error('❌ 画像生成API呼び出し失敗:', response.status);
				return null;
			}

			const result = await response.json() as {
				success: boolean;
				imageUrl?: string;
				message?: string;
			};

			if (result.success && result.imageUrl) {
				console.log('✅ 画像生成成功:', result.imageUrl);
				return result.imageUrl;
			} else {
				console.error('❌ 画像生成失敗:', result.message);
				return null;
			}
		} catch (error) {
			console.error('❌ 画像生成エラー:', error);
			return null;
		}
	}
}

/**
 * 評価統合機能: 3つの評価結果をAIEvaluationResponse形式で統合
 */
class EvaluationIntegrator {
	constructor(private workersai: ReturnType<typeof createWorkersAI>) { }

	async integrateEvaluations(results: EvaluationResult[], generatedImageUrl?: string): Promise<AIEvaluationResponse> {
		const businessFeasibility = results.find(r => r.criterion === 'business_feasibility');
		const businessValue = results.find(r => r.criterion === 'business_value');
		const technicalValidity = results.find(r => r.criterion === 'technical_validity');

		const totalScore = results.reduce((sum, result) => sum + result.score, 0);

		// 総合コメントを生成
		const comments = results.filter(r => r.reason).map(r => r.reason);
		const comment = comments.length > 0
			? `総合評価: ${comments.join('、')}`
			: '評価が完了しました。';

		console.log('🎯 評価結果統合完了');

		return {
			totalScore,
			comment,
			generatedImageUrl,
			breakdown: {
				criteria1: businessFeasibility?.score || 0,
				criteria2: businessValue?.score || 0,
				criteria3: technicalValidity?.score || 0,
			}
		};
	}

}

export class MyAgent extends Agent<Env> {
	async onRequest(request: Request) {
		try {
			const data = await request.json<AIEvaluationRequest | { prompt?: string }>();

			// AIEvaluationRequest形式の評価リクエスト
			if ('theme' in data && 'direction' in data && 'idea' in data && 'techNames' in data) {
				const evaluationData = data as AIEvaluationRequest;
				const result = await this.evaluateHackathonIdea(evaluationData);
				return Response.json(result);
			} 
			// 従来のプロンプト形式（後方互換性）
			else if ('prompt' in data && data.prompt) {
				const stream = await this.callAIModel(data.prompt);
				return stream.toTextStreamResponse({
					headers: {
						'Content-Type': 'text/x-unknown',
						'content-encoding': 'identity',
						'transfer-encoding': 'chunked',
					},
				});
			} else {
				return Response.json({
					error: 'Invalid request. Provide AIEvaluationRequest format: { theme, direction, idea, techNames } for evaluation, or { prompt } for simple AI chat'
				}, { status: 400 });
			}
		} catch (error) {
			return Response.json({
				error: 'Failed to process request',
				details: error instanceof Error ? error.message : 'Unknown error'
			}, { status: 500 });
		}
	}

	/**
	 * ハッカソンAI審査員による評価を実行
	 */
	async evaluateHackathonIdea(request: AIEvaluationRequest): Promise<AIEvaluationResponse> {
		const workersai = createWorkersAI({ binding: this.env.AI });

		try {
			// リクエストから必要な情報を取得
			const { theme, direction, idea, techNames } = request;
			
			// 1. オーケストレーター: 3つの評価タスクを作成
			const orchestrator = new EvaluationOrchestrator(workersai);
			const tasks = orchestrator.createEvaluationTasks(idea, techNames.join(', '), theme, direction);

			// 2. 並列実行: AI評価と画像生成を同時実行
			const worker = new EvaluationWorker(workersai);
			const imageGenerator = new ImageGenerator(this.env.IMAGE_API_URL);

			console.log(`🎯 ${tasks.length}個の評価タスクと画像生成を並列実行開始...`);
			const startTime = Date.now();

			// AI評価と画像生成を並列実行
			const [results, generatedImageUrl] = await Promise.all([
				// AI評価タスクの並列実行
				Promise.all(
					tasks.map(async (task, index) => {
						console.log(`📊 評価ワーカー${index + 1}: ${task.description} の評価開始`);
						const taskStartTime = Date.now();
						const result = await worker.evaluateTask(task, idea, techNames.join(', '), theme, direction);
						const taskEndTime = Date.now();
						console.log(`✅ 評価ワーカー${index + 1}: ${task.criterion} 完了 (${result.score}/${task.maxScore}点, ${taskEndTime - taskStartTime}ms)`);
						return result;
					})
				),
				// 画像生成の並列実行
				imageGenerator.generateImage(idea)
			]);

			const endTime = Date.now();
			console.log(`🎉 全処理完了: ${endTime - startTime}ms`);

			// 3. 統合: 評価結果と画像URLを指定フォーマットで統合
			const integrator = new EvaluationIntegrator(workersai);
			const finalEvaluation = await integrator.integrateEvaluations(results, generatedImageUrl || undefined);

			return finalEvaluation;
		} catch (error) {
			// フォールバック: エラー時の評価結果を返す
			console.error('⚠️ 評価処理でエラーが発生:', error);

			const fallbackEvaluation: AIEvaluationResponse = {
				totalScore: 0,
				comment: '評価中にエラーが発生しました',
				breakdown: {
					criteria1: 0,    // ビジネス的実現性（20点満点）
					criteria2: 0,    // ビジネス的価値（20点満点）
					criteria3: 0,    // 技術の妥当性（20点満点）
				}
			};

			return fallbackEvaluation;
		}
	}

	async callAIModel(prompt: string) {
		const workersai = createWorkersAI({ binding: this.env.AI });
		const model = workersai("@cf/deepseek-ai/deepseek-r1-distill-qwen-32b");

		return streamText({
			model: model,
			prompt: prompt,
		});
	}
}

export default {
	async fetch(request: Request, env: Env) {
		// --- CORS setup - Allow all origins ---
		const corsHeaders = new Headers({
			'Access-Control-Allow-Origin': '*',
			'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
			'Access-Control-Allow-Headers': 'Content-Type, Authorization',
			'Access-Control-Max-Age': '86400',
		});

		// Preflight request
		if (request.method === 'OPTIONS') {
			return new Response(null, { status: 204, headers: corsHeaders });
		}

		const agentId = new URL(request.url).searchParams.get('agent-id') || 'default-agent';
		const agent = await getAgentByName<Env, MyAgent>(env.MyAgent, agentId);
		const response = await agent.fetch(request);

		// Attach CORS headers to actual response
		// Clone to ensure headers are mutable and preserved
		const newHeaders = new Headers(response.headers);
		corsHeaders.forEach((value, key) => newHeaders.set(key, value));
		return new Response(response.body, {
			status: response.status,
			statusText: response.statusText,
			headers: newHeaders,
		});
	},
};
