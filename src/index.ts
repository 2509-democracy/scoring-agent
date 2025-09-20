/**
 * Hackathon AI Judge - Cloudflare Agents with Orchestrator-Workers Pattern
 *
 * This worker implements an AI-powered hackathon judging system using the Orchestrator-Workers pattern:
 * - Orchestrator: Creates 4 evaluation tasks (theme match, originality, tech compatibility, feasibility)
 * - Workers: Each worker uses a different AI model to evaluate one specific criterion
 * - Integration: Combines all evaluations into a standardized score format
 *
 * Usage:
 * POST request with JSON body: { "idea": "your idea", "technology": "tech stack", "theme": "optional theme" }
 * Returns: Formatted evaluation with SCORE: [total] and detailed breakdown
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
}

// 評価タスクのスキーマ
const EvaluationTaskSchema = z.object({
	taskId: z.string(),
	criterion: z.enum(['theme_match', 'originality', 'tech_compatibility', 'feasibility']),
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
	themeMatch: z.object({
		score: z.number(),
		reason: z.string(),
	}),
	originality: z.object({
		score: z.number(),
		reason: z.string(),
	}),
	techCompatibility: z.object({
		score: z.number(),
		reason: z.string(),
	}),
	feasibility: z.object({
		score: z.number(),
		reason: z.string(),
	}),
});

// 型定義
type EvaluationTask = z.infer<typeof EvaluationTaskSchema>;
type EvaluationResult = z.infer<typeof EvaluationResultSchema>;
type FinalEvaluation = z.infer<typeof FinalEvaluationSchema>;

interface HackathonEvaluationResult {
	tasks: EvaluationTask[];
	results: EvaluationResult[];
	finalScore: string;
}

/**
 * 評価オーケストレーター: 4つの評価観点のタスクを作成・配信
 */
class EvaluationOrchestrator {
	constructor(private workersai: ReturnType<typeof createWorkersAI>) { }

	createEvaluationTasks(idea: string, technology: string, theme?: string): EvaluationTask[] {
		const timestamp = Date.now();

		return [
			{
				taskId: `theme-match-${timestamp}`,
				criterion: 'theme_match' as const,
				maxScore: 20,
				description: 'テーマへの合致度（20点）: アイデアがテーマに沿っているか',
			},
			{
				taskId: `originality-${timestamp}`,
				criterion: 'originality' as const,
				maxScore: 30,
				description: 'アイデアの独創性（30点）: 他と差別化できるユニークさ',
			},
			{
				taskId: `tech-compatibility-${timestamp}`,
				criterion: 'tech_compatibility' as const,
				maxScore: 20,
				description: 'アイデアと技術の親和性（20点）: アイデアと技術の組み合わせの適切さ',
			},
			{
				taskId: `feasibility-${timestamp}`,
				criterion: 'feasibility' as const,
				maxScore: 30,
				description: '実現可能性（30点）: 現実的な実装が可能か',
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
		switch (criterion) {
			case 'theme_match':
				// テーマへの合致度: テーマ理解に優れたモデル
				return this.workersai("@cf/meta/llama-3.1-8b-instruct");
			case 'originality':
				// アイデアの独創性: 創造性評価に優れたモデル
				return this.workersai("@cf/google/gemma-3-12b-it");
			case 'tech_compatibility':
				// 技術親和性: 技術理解に優れたモデル
				return this.workersai("@cf/qwen/qwen2.5-coder-32b-instruct");
			case 'feasibility':
				// 実現可能性: 実装可能性判断に優れたモデル
				return this.workersai("@cf/meta/llama-3.2-3b-instruct");
			default:
				return this.workersai("@cf/meta/llama-3.1-8b-instruct");
		}
	}

	/**
	 * 評価観点別のシステムプロンプトを取得
	 */
	private getSystemPromptForCriterion(criterion: string): string {
		const basePrompt = 'あなたはハッカソンの専門審査員です。公正で客観的な評価を行ってください。レスポンスは必ず有効なJSON形式のみで返してください。';

		switch (criterion) {
			case 'theme_match':
				return `${basePrompt} テーマへの合致度を専門的に評価してください。テーマとアイデアの関連性、テーマの本質的な理解度を重視してください。`;
			case 'originality':
				return `${basePrompt} アイデアの独創性を専門的に評価してください。新規性、ユニークさ、既存アイデアとの差別化を重視してください。`;
			case 'tech_compatibility':
				return `${basePrompt} アイデアと技術の親和性を専門的に評価してください。技術選択の適切さ、実装の効率性を重視してください。`;
			case 'feasibility':
				return `${basePrompt} 実現可能性を専門的に評価してください。技術的実装可能性、リソース要件、開発期間を重視してください。`;
			default:
				return basePrompt;
		}
	}

	async evaluateTask(task: EvaluationTask, idea: string, technology: string, theme?: string): Promise<EvaluationResult> {
		try {
			const model = this.selectModelForCriterion(task.criterion);
			const systemPrompt = this.getSystemPromptForCriterion(task.criterion);

			const themeContext = theme ? `テーマ: ${theme}\n` : '';

			const evaluationStream = await streamText({
				model: model,
				system: `${systemPrompt} 必ず以下のJSON形式のみで回答してください。前置きや説明文、思考過程は一切含めず、純粋なJSONオブジェクトのみを出力してください。`,
				prompt: `${themeContext}アイデア: ${idea}
使用技術: ${technology}

評価観点: ${task.description}
最大点: ${task.maxScore}点

重要: 以下のJSON形式のみで回答してください。他のテキストは一切含めないでください:

{"score": [0-${task.maxScore}の整数], "reason": "評価理由を具体的に説明"}`,
			});

			const evaluationText = await evaluationStream.text;
			console.log(`🔍 AI Response for ${task.criterion}:`, evaluationText.substring(0, 200) + '...');

			// 複数の方法でJSONを抽出を試行
			let evaluation = null;
			let extractedJson = '';

			// 方法1: 最初の完全なJSONオブジェクトを抽出
			const jsonMatches = evaluationText.match(/\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}/g);
			if (jsonMatches && jsonMatches.length > 0) {
				for (const match of jsonMatches) {
					try {
						evaluation = JSON.parse(match);
						extractedJson = match;
						break;
					} catch (e) {
						continue;
					}
				}
			}

			// 方法2: scoreとreasonを直接抽出
			if (!evaluation) {
				const scoreMatch = evaluationText.match(/"score"\s*:\s*(\d+)/);
				const reasonMatch = evaluationText.match(/"reason"\s*:\s*"([^"]+)"/);

				if (scoreMatch && reasonMatch) {
					evaluation = {
						score: parseInt(scoreMatch[1]),
						reason: reasonMatch[1]
					};
					extractedJson = `{"score": ${scoreMatch[1]}, "reason": "${reasonMatch[1]}"}`;
				}
			}

			// 方法3: 数値とテキストを正規表現で抽出
			if (!evaluation) {
				const scorePattern = /(?:score|点数|評価)[:\s]*(\d+)/i;
				const reasonPattern = /(?:reason|理由)[:\s]*["']?([^"'\n]{10,})["']?/i;

				const scoreMatch = evaluationText.match(scorePattern);
				const reasonMatch = evaluationText.match(reasonPattern);

				if (scoreMatch) {
					evaluation = {
						score: parseInt(scoreMatch[1]),
						reason: reasonMatch ? reasonMatch[1].trim() : `${task.criterion}の評価を実行しました`
					};
				}
			}

			// 方法4: 空のscoreフィールドを修正
			if (!evaluation) {
				const brokenJsonMatch = evaluationText.match(/\{"score":\s*,\s*"reason":\s*"([^"]+)"\}/);
				if (brokenJsonMatch) {
					evaluation = {
						score: Math.floor(task.maxScore * 0.6), // デフォルト60%
						reason: brokenJsonMatch[1]
					};
				}
			}

			// 方法5: 思考タグを除去してから再試行
			if (!evaluation && evaluationText.includes('<think>')) {
				const cleanText = evaluationText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
				const scoreMatch = cleanText.match(/(?:score|点数)[:\s]*(\d+)/i);
				const reasonMatch = cleanText.match(/(?:reason|理由)[:\s]*["']?([^"'\n]{10,})["']?/i);

				if (scoreMatch) {
					evaluation = {
						score: parseInt(scoreMatch[1]),
						reason: reasonMatch ? reasonMatch[1].trim() : `${task.criterion}の評価を実行しました`
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
 * 評価統合機能: 4つの評価結果を指定フォーマットで統合
 */
class EvaluationIntegrator {
	constructor(private workersai: ReturnType<typeof createWorkersAI>) { }

	async integrateEvaluations(results: EvaluationResult[]): Promise<string> {
		try {
			const themeMatch = results.find(r => r.criterion === 'theme_match');
			const originality = results.find(r => r.criterion === 'originality');
			const techCompatibility = results.find(r => r.criterion === 'tech_compatibility');
			const feasibility = results.find(r => r.criterion === 'feasibility');


			const totalScore = results.reduce((sum, result) => sum + result.score, 0);

			const model = this.workersai("@cf/meta/llama-3.2-3b-instruct");

			const summaryPrompt = `あなたは評価結果をまとめる専門家です。以下の評価結果を必ず指定された形式でまとめてください。

評価結果:
- テーマへの合致度: ${themeMatch?.score || 0}/20点 理由: ${themeMatch?.reason || '評価エラー'}
- アイデアの独創性: ${originality?.score || 0}/30点 理由: ${originality?.reason || '評価エラー'}
- アイデアと技術の親和性: ${techCompatibility?.score || 0}/20点 理由: ${techCompatibility?.reason || '評価エラー'}
- 実現可能性: ${feasibility?.score || 0}/30点 理由: ${feasibility?.reason || '評価エラー'}

合計点: ${totalScore}点

**重要: 以下の形式で必ず出力してください。この形式以外は絶対に使用しないでください:**

SCORE: [合計点]
        テーマへの合致度: [点数]/20点 理由: [理由]
        アイデアの独創性: [点数]/30点 理由: [理由]
        アイデアと技術の親和性: [点数]/20点 理由: [理由]
        実現可能性: [点数]/30点 理由: [理由]

この形式を厳密に守って出力してください。前置きや説明は一切不要です。`;

			const summaryStream = await streamText({
				model: model,
				system: "あなたは評価結果をまとめる専門家です。指定された形式を厳密に守って出力してください。前置きや説明文は一切含めず、指定された形式のみで回答してください。",
				prompt: summaryPrompt,
			});

			const summaryText = await summaryStream.text;

			// AIの出力から指定形式を抽出
			const scorePattern = /SCORE:\s*(\d+)/;
			const themePattern = /テーマへの合致度:\s*(\d+)\/20点\s*理由:\s*(.+?)(?=\n|アイデアの独創性|$)/;
			const originalityPattern = /アイデアの独創性:\s*(\d+)\/30点\s*理由:\s*(.+?)(?=\n|アイデアと技術|$)/;
			const techPattern = /アイデアと技術の親和性:\s*(\d+)\/20点\s*理由:\s*(.+?)(?=\n|実現可能性|$)/;
			const feasibilityPattern = /実現可能性:\s*(\d+)\/30点\s*理由:\s*(.+?)(?=\n|$)/;

			// パターンマッチングで抽出
			const scoreMatch = summaryText.match(scorePattern);
			const themeMatchResult = summaryText.match(themePattern);
			const originalityMatchResult = summaryText.match(originalityPattern);
			const techMatchResult = summaryText.match(techPattern);
			const feasibilityMatchResult = summaryText.match(feasibilityPattern);

			// 抽出できた場合はそれを使用、できなかった場合は元の値を使用
			const finalScore = scoreMatch ? scoreMatch[1] : totalScore.toString();
			const finalThemeScore = themeMatchResult ? themeMatchResult[1] : (themeMatch?.score || 0).toString();
			const finalThemeReason = themeMatchResult ? themeMatchResult[2].trim() : (themeMatch?.reason || '評価エラー');
			const finalOriginalityScore = originalityMatchResult ? originalityMatchResult[1] : (originality?.score || 0).toString();
			const finalOriginalityReason = originalityMatchResult ? originalityMatchResult[2].trim() : (originality?.reason || '評価エラー');
			const finalTechScore = techMatchResult ? techMatchResult[1] : (techCompatibility?.score || 0).toString();
			const finalTechReason = techMatchResult ? techMatchResult[2].trim() : (techCompatibility?.reason || '評価エラー');
			const finalFeasibilityScore = feasibilityMatchResult ? feasibilityMatchResult[1] : (feasibility?.score || 0).toString();
			const finalFeasibilityReason = feasibilityMatchResult ? feasibilityMatchResult[2].trim() : (feasibility?.reason || '評価エラー');

			// 指定フォーマットで確実に出力
			const formattedResult = this.formatEvaluationResult(
				finalScore,
				finalThemeScore,
				finalThemeReason,
				finalOriginalityScore,
				finalOriginalityReason,
				finalTechScore,
				finalTechReason,
				finalFeasibilityScore,
				finalFeasibilityReason
			);

			console.log('🎯 最終評価結果:', formattedResult);
			return formattedResult;

		} catch (error) {
			// エラー時のフォールバック: 元の形式で出力
			console.warn('⚠️ 統合処理でエラーが発生、フォールバック出力を使用:', error);

			const themeMatch = results.find(r => r.criterion === 'theme_match');
			const originality = results.find(r => r.criterion === 'originality');
			const techCompatibility = results.find(r => r.criterion === 'tech_compatibility');
			const feasibility = results.find(r => r.criterion === 'feasibility');
			const totalScore = results.reduce((sum, result) => sum + result.score, 0);

			return this.formatEvaluationResult(
				totalScore,
				themeMatch?.score || 0,
				themeMatch?.reason || '評価エラー',
				originality?.score || 0,
				originality?.reason || '評価エラー',
				techCompatibility?.score || 0,
				techCompatibility?.reason || '評価エラー',
				feasibility?.score || 0,
				feasibility?.reason || '評価エラー'
			);
		}
	}

	/**
	 * スコア評価結果を指定フォーマットで整形する
	 */
	private formatEvaluationResult(
		totalScore: string | number,
		themeScore: string | number,
		themeReason: string,
		originalityScore: string | number,
		originalityReason: string,
		techScore: string | number,
		techReason: string,
		feasibilityScore: string | number,
		feasibilityReason: string
	): string {
		return `SCORE: ${totalScore}
        テーマへの合致度: ${themeScore}/20点 理由: ${themeReason}
        アイデアの独創性: ${originalityScore}/30点 理由: ${originalityReason}
        アイデアと技術の親和性: ${techScore}/20点 理由: ${techReason}
        実現可能性: ${feasibilityScore}/30点 理由: ${feasibilityReason}`;
	}
}

export class MyAgent extends Agent<Env> {
	async onRequest(request: Request) {
		try {
			const data = await request.json<{
				idea?: string;
				technology?: string;
				theme?: string;
				prompt?: string;
			}>();

			if (data.idea && data.technology) {
				const result = await this.evaluateHackathonIdea(data.idea, data.technology, data.theme);
				return Response.json(result);
			} else if (data.prompt) {
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
					error: 'Invalid request. Provide "idea" and "technology" for evaluation, or "prompt" for simple AI chat'
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
	async evaluateHackathonIdea(idea: string, technology: string, theme?: string): Promise<HackathonEvaluationResult> {
		const workersai = createWorkersAI({ binding: this.env.AI });

		try {
			// 1. オーケストレーター: 4つの評価タスクを作成
			const orchestrator = new EvaluationOrchestrator(workersai);
			const tasks = orchestrator.createEvaluationTasks(idea, technology, theme);

			// 2. ワーカー: 4つの評価観点を並列実行
			const worker = new EvaluationWorker(workersai);
			console.log(`🎯 ${tasks.length}個の評価タスクを並列実行開始...`);
			const startTime = Date.now();

			const results = await Promise.all(
				tasks.map(async (task, index) => {
					console.log(`📊 評価ワーカー${index + 1}: ${task.description} の評価開始`);
					const taskStartTime = Date.now();
					const result = await worker.evaluateTask(task, idea, technology, theme);
					const taskEndTime = Date.now();
					console.log(`✅ 評価ワーカー${index + 1}: ${task.criterion} 完了 (${result.score}/${task.maxScore}点, ${taskEndTime - taskStartTime}ms)`);
					return result;
				})
			);

			const endTime = Date.now();
			console.log(`🎉 全評価完了: ${endTime - startTime}ms`);

			// 3. 統合: 評価結果を指定フォーマットで統合
			const integrator = new EvaluationIntegrator(workersai);
			const finalScore = await integrator.integrateEvaluations(results);

			return {
				tasks,
				results,
				finalScore,
			};
		} catch (error) {
			// フォールバック: エラー時の評価結果を返す
			const fallbackTasks: EvaluationTask[] = [
				{ taskId: "fallback-theme", criterion: "theme_match", maxScore: 20, description: "テーマへの合致度" },
				{ taskId: "fallback-originality", criterion: "originality", maxScore: 30, description: "アイデアの独創性" },
				{ taskId: "fallback-tech", criterion: "tech_compatibility", maxScore: 20, description: "技術親和性" },
				{ taskId: "fallback-feasibility", criterion: "feasibility", maxScore: 30, description: "実現可能性" }
			];

			const fallbackResults: EvaluationResult[] = fallbackTasks.map(task => ({
				taskId: task.taskId,
				criterion: task.criterion,
				score: 0,
				maxScore: task.maxScore,
				reason: '評価中にエラーが発生しました。',
				success: false,
				error: error instanceof Error ? error.message : 'Unknown error'
			}));

			const fallbackScore = `SCORE: 0
テーマへの合致度: 0/20点 理由: 評価中にエラーが発生しました
アイデアの独創性: 0/30点 理由: 評価中にエラーが発生しました
アイデアと技術の親和性: 0/20点 理由: 評価中にエラーが発生しました
実現可能性: 0/30点 理由: 評価中にエラーが発生しました`;

			return {
				tasks: fallbackTasks,
				results: fallbackResults,
				finalScore: fallbackScore,
			};
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
		// --- CORS setup ---
		const origin = request.headers.get('Origin');
		const configuredOrigins = (env.CORS_ALLOWED_ORIGINS || '')
			.split(',')
			.map((s) => s.trim())
			.filter((s) => s.length > 0);
		const allowedOrigins = new Set<string>([
			...configuredOrigins,
		]);

		const isAllowedOrigin = origin != null && allowedOrigins.has(origin);

		const corsHeaders = new Headers({
			'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
			'Access-Control-Allow-Headers': 'Content-Type, Authorization',
			'Access-Control-Max-Age': '86400',
			'Vary': 'Origin',
		});

		if (isAllowedOrigin && origin) {
			corsHeaders.set('Access-Control-Allow-Origin', origin);
		}

		// Preflight request
		if (request.method === 'OPTIONS') {
			// Handle CORS preflight
			if (!isAllowedOrigin) {
				return new Response('CORS origin not allowed', { status: 403, headers: corsHeaders });
			}
			return new Response(null, { status: 204, headers: corsHeaders });
		}

		// Block disallowed cross-origin requests (when Origin header is present)
		if (origin && !isAllowedOrigin) {
			return new Response('CORS origin not allowed', { status: 403, headers: corsHeaders });
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
