{{/*
Common labels
*/}}
{{- define "air-bench.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
vLLM fullname
*/}}
{{- define "air-bench.vllm.fullname" -}}
{{ .Release.Name }}-vllm
{{- end }}

{{/*
Benchmark fullname
*/}}
{{- define "air-bench.benchmark.fullname" -}}
{{ .Release.Name }}-benchmark
{{- end }}
